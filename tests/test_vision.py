"""Tools for the champ-llm local training pipeline."""
import json
import io
from vision_integrity import sha256_file
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from prepare_vision import prepare_vision, hash_split
from vision_model import VisionEncoder, image_to_tensor, multimodal_loss, iter_examples
from capture_website import verify_url, load_interactions, capture_sections, project_name_for_url
from compare_screenshots import compare
import prepare_data as text_prep


class TinyDummyLM(nn.Module):
    def __init__(self, hidden=32, vocab=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, *, inputs_embeds, attention_mask, labels):
        
        running = inputs_embeds.cumsum(dim=1) / torch.arange(1, inputs_embeds.shape[1]+1, device=inputs_embeds.device)[None, :, None]
        logits = self.head(running)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        return type('Reply', (), {'loss': loss})()


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / 'training_data'
        self.sources.mkdir()
        self.out = self.root / 'outputs'

    def tearDown(self):
        self.temp.cleanup()

    def add_image(self, project, name='page.png', annotation=None):
        p = self.sources / project / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (128, 96), 'white').save(p)
        if annotation is not None:
            annotation['image_sha256'] = sha256_file(p)
            p.with_suffix('.vision.json').write_text(json.dumps(annotation, ensure_ascii=False), encoding='utf-8')
        return p

    def test_prepare_pair_and_code(self):
        source = self.sources / 'programming' / 'react' / 'Button.jsx'
        source.parent.mkdir(parents=True)
        source.write_text('export default function Bouton() { return <button>OK</button> }', encoding='utf-8')
        image = self.add_image('project-a', annotation={
            'project': 'a', 'description': 'En-tête minimal', 'critique': 'Contraste trop faible',
            'code_files': ['programming/react/Button.jsx'],
        })
        report = prepare_vision(self.sources, self.out, validation=0, root=self.root)
        self.assertEqual(1, report['annotated_images'])
        rows = list(iter_examples(self.out/'vision'/'vision_manifest.json', self.root))
        self.assertEqual(1, len(rows))
        self.assertEqual(image.resolve(), rows[0][0])
        self.assertIn('Contraste trop faible', rows[0][1]['answer'])
        self.assertIn('export default', rows[0][1]['answer'])
        self.assertEqual([], report['skipped_images'])

    def test_no_annotation_and_draft_not_trained(self):
        image = self.add_image('test')
        report = prepare_vision(self.sources, self.out, root=self.root)
        self.assertEqual(0, report['annotated_images'])
        self.assertIn(image.relative_to(self.root).as_posix(), report['unannotated_images'])

    def test_partition_by_project(self):
        for i in range(10):
            self.add_image('meme-project', name=f'page-{i}.png', annotation={
                'project':'stable-project', 'critique':f'Problème {i}',
            })
        prepare_vision(self.sources, self.out, validation=.2, root=self.root)
        manifest = json.loads((self.out/'vision'/'vision_manifest.json').read_text(encoding='utf-8'))
        splits = manifest['splits']
        count = [splits[x]['records'] for x in ('vision_train', 'vision_validation')]
        self.assertEqual(sum(count), 10)
        self.assertIn(0, count)  
        self.assertEqual(hash_split('stable-project', .2), hash_split('stable-project', .2))

    def test_outside_code_path_is_rejected(self):
        self.add_image('fixture-site', annotation={'project':'fixture-site', 'description':'Visible', 'code_files':['../../secrets.txt']})
        report = prepare_vision(self.sources, self.out, validation=0, root=self.root)
        self.assertEqual(1, report['annotated_images'])
        self.assertTrue(report['warnings'])
        row = list(iter_examples(self.out/'vision'/'vision_manifest.json', self.root))[0][1]
        self.assertNotIn('Associated code', row['answer'])

    def test_visual_bridge_has_gradient(self):
        llm = TinyDummyLM(hidden=32)
        bridge = VisionEncoder(32, image_size=32, patch_size=16, vision_dim=32, vision_layers=1)
        pixels = torch.rand(1, 3, 32, 32)
        ids = torch.tensor([[4,5,6,7,8,9]])
        labels = torch.tensor([[-100,-100,-100,7,8,9]])
        loss = multimodal_loss(llm, bridge, pixels, ids, labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertGreater(bridge.projection_patch.weight.grad.abs().sum().item(), 0)

    def test_integrated_preparation_separates_image_annotations_from_text(self):
        original = (text_prep.ROOT, text_prep.SOURCES, text_prep.SINGLE, text_prep.OUTPUT)
        text_prep.ROOT = self.root
        text_prep.SOURCES = self.sources
        text_prep.SINGLE = self.root/'training_data.txt'
        text_prep.OUTPUT = self.out
        try:
            (self.sources/'parler.txt').write_text('Salut, je veux étudier le HTML.', encoding='utf-8')
            self.add_image('portfolio', annotation={'project':'portfolio','critique':'Problème de contraste'})
            text_prep.main([])
            data = json.loads((self.out/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(1, data['pretrain_documents'])
            vision = json.loads((self.out/'vision'/'vision_report.json').read_text(encoding='utf-8'))
            self.assertEqual(1, vision['annotated_images'])
            self.assertEqual(1, len(list(iter_examples(self.out/'vision'/'vision_manifest.json', self.root,
                                                    split=hash_split('portfolio', .05)))))
        finally:
            text_prep.ROOT, text_prep.SOURCES, text_prep.SINGLE, text_prep.OUTPUT = original

    def test_pixel_comparison_and_resize(self):
        a = self.add_image('tests', name='before.png')
        b = self.sources/'tests'/'after.png'
        Image.new('RGB', (128,96), 'red').save(b)
        result = compare(a, b, self.out/'difference.png')
        self.assertGreater(result['mean_pixel_difference_pct'], 0)
        self.assertTrue((self.out/'difference.png').is_file())
        tensor = image_to_tensor(a, 32)
        self.assertEqual((3,32,32), tuple(tensor.shape))

    def test_automatic_dom_section_captures_and_annotation_preservation(self):
        class Element:
            def __init__(self, tag, w, h, visible=True):
                self.tag, self.w, self.h, self.visible = tag, w, h, visible
            def is_visible(self):
                return self.visible
            def bounding_box(self):
                return {'width': self.w, 'height': self.h} if self.visible else None
            def evaluate(self, js):
                return {'tag': self.tag, 'role': '', 'id': '', 'classes': '', 'title': 'Démo'}
            def screenshot(self, *, animations, timeout):
                buffer = io.BytesIO()
                Image.new('RGB', (180, 100), 'white').save(buffer, format='PNG')
                return buffer.getvalue()
        class Locator:
            def __init__(self):
                self.items = [Element('header', 400, 100), Element('section', 400, 30),
                              Element('section', 400, 200), Element('footer', 400, 80)]
            def count(self):
                return len(self.items)
            def nth(self, i):
                return self.items[i]
        class Page:
            def locator(self, css):
                self.selector = css
                return Locator()
        folder = self.sources / 'vision' / 'auto'
        folder.mkdir(parents=True)
        result = capture_sections(Page(), folder, project='auto', viewport_label='desktop',
                                  viewport_width=1440, max_sections=3)
        self.assertEqual(3, len(result))
        self.assertTrue(all('capture' in x for x in result))
        sidecar = (folder / result[0]['capture']).with_suffix('.vision.json')
        original = json.loads(sidecar.read_text(encoding='utf-8'))
        self.assertEqual(original['detected_element']['tag'], 'header')
        original['critique'] = 'Ne pas écraser mes notes'
        sidecar.write_text(json.dumps(original,ensure_ascii=False),encoding='utf8')
        result2 = capture_sections(Page(), folder, project='auto', viewport_label='desktop',
                                   viewport_width=1440, max_sections=3)
        self.assertEqual(result, result2)
        self.assertEqual(json.loads(sidecar.read_text(encoding='utf8'))['critique'], 'Ne pas écraser mes notes')
        self.assertEqual([], capture_sections(Page(), folder, project='auto',
                                              viewport_label='mobile', viewport_width=375, max_sections=0))
        self.assertEqual(project_name_for_url('https://exemple.com/accueil'),
                         project_name_for_url('https://exemple.com/accueil'))
        self.assertNotEqual(project_name_for_url('https://exemple.com/accueil'),
                            project_name_for_url('https://exemple.com/contact'))

    def test_local_url_guards_and_interaction_format(self):
        verify_url('http://localhost:3000', False)
        with self.assertRaises(ValueError):
            verify_url('https://example.com', False)
        verify_url('https://example.com', True)
        file = self.root/'tests.json'
        file.write_text('{"scenarios":[{"selector":"#menu","action":"click","expected_visible":"nav"}]}')
        self.assertEqual('click', load_interactions(file)[0]['action'])


if __name__ == '__main__':
    unittest.main()
