"""Tools for the champ-llm local training pipeline."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from capture_website import create_annotation, capture_sections
from prepare_vision import prepare_vision
from vision_integrity import (save_capture, sha256_file,
                              verify_vision_record)
from vision_model import iter_examples
from bind_annotation import bind


def png(couleur):
    out = io.BytesIO()
    Image.new('RGB', (150, 110), couleur).save(out, format='PNG')
    return out.getvalue()


class BoundImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / 'training_data' / 'vision' / 'fixture-site'
        self.folder.mkdir(parents=True)
        self.outputs = self.root / 'outputs'

    def tearDown(self):
        self.temp.cleanup()

    def test_changed_screenshot_gets_new_annotation(self):
        a, ha = save_capture(self.folder, 'desktop-1440', png('red'))
        ann_a = create_annotation(a, project='fixture-site', width=1440)
        content = json.loads(ann_a.read_text())
        self.assertEqual(content['image_sha256'], ha)
        content['critique'] = 'Titre trop petit sur la version rouge'
        ann_a.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')
        b, hb = save_capture(self.folder, 'desktop-1440', png('blue'))
        ann_b = create_annotation(b, project='fixture-site', width=1440)
        self.assertNotEqual(a, b)
        self.assertNotEqual(ha, hb)
        self.assertNotEqual(ann_a, ann_b)
        self.assertEqual(json.loads(ann_b.read_text())['critique'], '')
        self.assertEqual(json.loads(ann_a.read_text())['critique'], content['critique'])
        a2, _ = save_capture(self.folder, 'desktop-1440', png('red'))
        self.assertEqual(a, a2)
        self.assertEqual(create_annotation(a2, project='fixture-site', width=1440), ann_a)

    def test_modified_image_rejected_during_training(self):
        a, _ = save_capture(self.folder, 'mobile-375', png('red'))
        ann = create_annotation(a, project='fixture-site', width=375)
        data = json.loads(ann.read_text())
        data['critique'] = 'Bouton trop près du bord'
        ann.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        report = prepare_vision(self.root/'training_data', self.outputs, validation=0, root=self.root)
        self.assertEqual(report['annotated_images'], 1)
        manifest = self.outputs/'vision'/'vision_manifest.json'
        image, row = list(iter_examples(manifest, self.root))[0]
        self.assertEqual(image, a.resolve())
        self.assertEqual(row['image_sha256'], sha256_file(a))
        Image.new('RGB', (150,110), 'green').save(a)
        with self.assertRaisesRegex(ValueError, 'Image changed'):
            list(iter_examples(manifest, self.root))
        bad = prepare_vision(self.root/'training_data', self.outputs, validation=0, root=self.root)
        self.assertEqual(bad['annotated_images'], 0)
        self.assertTrue(bad['skipped_images'])

    def test_changed_annotation_rejected_after_preparation(self):
        a, _ = save_capture(self.folder, 'mobile-375', png('red'))
        ann = create_annotation(a, project='fixture-site', width=375)
        data = json.loads(ann.read_text()); data['critique'] = 'Texte peu lisible'
        ann.write_text(json.dumps(data),encoding='utf8')
        prepare_vision(self.root/'training_data', self.outputs, validation=0, root=self.root)
        manifest = self.outputs/'vision'/'vision_manifest.json'
        self.assertEqual(len(list(iter_examples(manifest, self.root))),1)
        data['critique'] = 'Nouveau comment'
        ann.write_text(json.dumps(data),encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Annotation changed'):
            list(iter_examples(manifest,self.root))

    def test_old_annotation_requires_human_review(self):
        a = self.folder/'ancienne.png'; a.write_bytes(png('white'))
        ann = a.with_suffix('.vision.json')
        ann.write_text(json.dumps({'critique':'Ancienne critique', 'project':'fixture-site'}),encoding='utf8')
        report = prepare_vision(self.root/'training_data',self.outputs,validation=0,root=self.root)
        self.assertEqual(report['annotated_images'],0)
        self.assertTrue(report['skipped_images'])
        with self.assertRaisesRegex(ValueError, 'Before binding'):
            bind(a, base=self.root/'training_data')
        bind(a, base=self.root/'training_data', confirm=True)
        report = prepare_vision(self.root/'training_data',self.outputs,validation=0,root=self.root)
        self.assertEqual(report['annotated_images'],1)
        Image.new('RGB',(150,110),'yellow').save(a)
        with self.assertRaisesRegex(ValueError, 'DIFFERENT image'):
            bind(a, base=self.root/'training_data', confirm=True)

    def test_dom_capture_never_reuses_a_stale_critique(self):
        class Element:
            color = 'red'
            def is_visible(self): return True
            def bounding_box(self): return {'width': 300,'height':110}
            def evaluate(self, _): return {'tag':'section','id':'hero','role':'','classes':'','title':'Titre'}
            def screenshot(self, *, animations, timeout): return png(self.color)
        class Locator:
            def count(self): return 1
            def nth(self, i): return self.elem
        class Page:
            def __init__(self):
                self.loc=Locator(); self.loc.elem=Element()
            def locator(self, _): return self.loc
        page=Page()
        first=capture_sections(page,self.folder,project='fixture-site',viewport_label='desktop',viewport_width=1440,max_sections=1)
        img1=self.folder/first[0]['capture']; ann1=img1.with_suffix('.vision.json')
        obj=json.loads(ann1.read_text());obj['critique']='Rouge problématique'
        ann1.write_text(json.dumps(obj),encoding='utf8')
        page.loc.elem.color='blue'
        second=capture_sections(page,self.folder,project='fixture-site',viewport_label='desktop',viewport_width=1440,max_sections=1)
        img2=self.folder/second[0]['capture'];ann2=img2.with_suffix('.vision.json')
        self.assertNotEqual(img1,img2)
        self.assertEqual(json.loads(ann2.read_text())['critique'],'')
        self.assertEqual(json.loads(ann1.read_text())['critique'],'Rouge problématique')


if __name__ == '__main__':
    unittest.main()
