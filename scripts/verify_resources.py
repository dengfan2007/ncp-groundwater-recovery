"""Verify a curated public resource checkout using only the Python standard library."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'resources_manifest.json').read_text(encoding='utf-8'))
    problems = []
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append('Missing or unsafe path: ' + item['path'])
        elif path.stat().st_size != item['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            problems.append('Changed: ' + item['path'])
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f"PASS: {len(manifest['files'])} resources verified by size and SHA-256.")


if __name__ == '__main__':
    main()
