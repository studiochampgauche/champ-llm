#!/usr/bin/env python3
"""Compare screenshots at the pixel level; this is not an aesthetic judgment."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageStat


def compare(before: Path, after: Path, output: Path) -> dict:
    with Image.open(before) as a, Image.open(after) as b:
        a = a.convert('RGB')
        b = b.convert('RGB')
        if a.size != b.size:
            raise ValueError('Screenshot dimensions differ: use the same viewport and page height.')
        diff = ImageChops.difference(a, b)
        stat = ImageStat.Stat(diff)
        avg = sum(stat.mean) / (3*255)
        output.parent.mkdir(parents=True, exist_ok=True)
        diff.save(output)
        return {'before': str(before), 'after': str(after), 'diff_image': str(output),
                'mean_pixel_difference_pct': round(avg*100, 2),
                'note': 'Pixel difference, not an assessment of UX or aesthetics.'}


def main():
    p = argparse.ArgumentParser(description='Compare two screenshots with Pillow.')
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/visual_analysis/difference.png'))
    args = p.parse_args()
    print(json.dumps(compare(args.before, args.after, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
