"""Check LMEF input layout using only the standard library; does not train."""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def class_counts(root, extensions, normalize=False):
    counts = Counter()
    for folder in sorted(root.iterdir()):
        if folder.is_dir():
            name = re.sub(r'^\d+', '', folder.name).strip().lower() if normalize else folder.name
            counts[name] += sum(p.is_file() and p.suffix.lower() in extensions
                                for p in folder.iterdir())
    return counts


def check(workflow, root):
    errors = []
    if not root.is_dir():
        return [f'Directory does not exist: {root}']
    if workflow == 'mendeley':
        for panel in (1, 2, 3):
            for condition in ('Clean', 'Dirt', 'Shadow'):
                for suffix in ('.csv', '.mat'):
                    path = root / f'Panel{panel}_{condition}{suffix}'
                    if not path.is_file() or path.stat().st_size == 0:
                        errors.append(f'Missing or empty input: {path.name}')
        if not errors:
            print('Found all 9 nonempty CSV/MAT pairs. Array contents have not been validated.')
        return errors
    extensions = IMAGE_EXTENSIONS | {'.tif', '.tiff'} if workflow == 'pvf10' else IMAGE_EXTENSIONS
    splits = ('train', 'test') if workflow == 'pvf10' else ('.',)
    labels = {}
    for split in splits:
        folder = root / split
        if not folder.is_dir():
            errors.append(f'Missing split folder: {folder}')
            continue
        counts = class_counts(folder, extensions, normalize=workflow == 'pvf10')
        labels[split] = set(counts)
        print(f'{split}: {sum(counts.values())} image files; {len(counts)} class folders after name mapping')
        for name, count in sorted(counts.items()):
            print(f'  {name}: {count}')
            if count == 0:
                errors.append(f'{split}/{name}: no supported images directly in class folder; check ZIP nesting and extensions')
            if not name:
                errors.append(f'{split}: empty class name after stripping numeric prefix')
        if len(counts) < 2:
            errors.append(f'{split}: classification requires at least two class folders')
        if workflow == 'thermal':
            small = sum(p.is_file() and p.suffix.lower() in extensions and p.stat().st_size <= 20000
                        for cls in folder.iterdir() if cls.is_dir() for p in cls.iterdir())
            print(f'{small} images will fail the general notebook byte-size rule (<= 20,000 bytes).')
    if workflow == 'pvf10' and len(labels) == 2 and labels['train'] != labels['test']:
        errors.append('Normalized train/test class names differ; inspect label mapping before training')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', choices=('pvf10', 'thermal', 'mendeley'))
    parser.add_argument('--root', required=True, type=Path, help='Input directory for the chosen workflow')
    args = parser.parse_args()
    try:
        errors = check(args.workflow, args.root.expanduser())
    except OSError as exc:
        errors = [f'Cannot inspect inputs: {exc}']
    for error in errors:
        print(f'ERROR: {error}', file=sys.stderr)
    if errors:
        return 1
    print('Layout check passed. File contents, split feasibility and scientific validity are not checked.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
