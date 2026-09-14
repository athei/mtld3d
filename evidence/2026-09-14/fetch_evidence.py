"""Download one issue's archived evidence at an immutable Git revision."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision', help='Full 40-character evidence commit SHA')
    parser.add_argument('issue', type=int, choices=(54, 510, 625, 638, 673))
    parser.add_argument('output', type=Path, help='New directory; existing paths are refused')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('revision must be a full lowercase commit SHA')
    args.output.mkdir(parents=True, exist_ok=False)
    base = f'https://raw.githubusercontent.com/athei/mtld3d/{args.revision}/evidence/2026-09-14/issue-{args.issue}/'
    with urlopen(base + 'ARCHIVE.json', timeout=60) as response:
        manifest_bytes = response.read()
    manifest = json.loads(manifest_bytes)
    (args.output / 'ARCHIVE.json').write_bytes(manifest_bytes)
    archive = args.output / f'issue-{args.issue}.tar.xz'
    checksum = hashlib.sha256()
    size = 0
    with archive.open('xb') as combined:
        for chunk in manifest['chunks']:
            if not re.fullmatch(rf'issue-{args.issue}\.tar\.xz\.part[0-9]{{2}}', chunk):
                raise ValueError('Unexpected archive chunk name')
            print(f'Downloading {chunk}', flush=True)
            with urlopen(base + chunk, timeout=60) as response:
                while data := response.read(1024 * 1024):
                    combined.write(data)
                    checksum.update(data)
                    size += len(data)
    if size != manifest['bytes'] or checksum.hexdigest() != manifest['sha256']:
        raise ValueError('Downloaded archive does not match its manifest; not extracting')
    with tarfile.open(archive, 'r:xz') as bundle:
        members = bundle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or path.parts[0] != f'issue-{args.issue}':
                raise ValueError(f'Unexpected archive member: {member.name}')
        bundle.extractall(args.output, members=members)
    payload = args.output / f'issue-{args.issue}'
    for line in (payload / 'PUBLIC-SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unexpected payload path')
        if hashlib.sha256((payload / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Payload checksum mismatch: {name}')
    print(f'Verified evidence: {payload}')


if __name__ == '__main__':
    main()
