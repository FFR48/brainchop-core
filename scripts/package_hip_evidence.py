#!/usr/bin/env python3
"""Create a deterministic public HIP evidence archive from text-only test outputs.

Never includes images, executable/code-object files, source checkouts or runtime
shims. Normalizes private paths, emails, process IDs, device addresses and CSV
clock origins while preserving numerical results and exact time differences.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

PRIVATE_PATH = re.compile(r'/(?:home|Users)/[^\s\x00\"\'<>;,)]*|/(?:private/)?(?:tmp|var/folders)/[^\s\x00\"\'<>;,)]*')
EMAIL = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
EXCLUDED = {'source', 'pr34-source', 'stubs', 'hidden-runtime', 'repacked', '__pycache__', 'tmp'}
SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|Authorization:\s*Bearer\s+\S+', re.I)


def sanitize_text(text, replacements):
    for source, token in sorted(replacements.items(), key=lambda item: -len(item[0])):
        text = text.replace(source, '${' + token + '}')
    text = PRIVATE_PATH.sub('${PRIVATE_PATH}', text)
    text = EMAIL.sub('${EMAIL}', text)
    text = re.sub(r'0x[0-9a-fA-F]{8,}', '${ADDRESS}', text)
    text = re.sub(r'(^[IWEF]\d{8} \S+)\s+\d+', r'\1 THREAD', text, flags=re.M)
    return text


def build_bundle(inputs, output, replacements):
    files, original_hashes = {}, {}
    csv_origins = {}
    selected = []
    for label, root in inputs.items():
        if not re.fullmatch(r'[a-z0-9-]+', label):
            raise ValueError('evidence label must contain lowercase letters, digits or hyphens')
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if path.is_symlink() or not path.is_file() or path.suffix not in ('.json', '.csv', '.log', '.time') or any(
                    part in EXCLUDED or part.startswith('.') for part in relative.parts):
                continue
            name = label + '/' + relative.as_posix()
            raw = path.read_bytes()
            text = raw.decode('utf-8')
            if SECRET.search(text):
                raise ValueError('possible credential in evidence: ' + name)
            selected.append((name, text))
            original_hashes[name] = hashlib.sha256(raw).hexdigest()
            if path.suffix == '.csv':
                for row in csv.DictReader(io.StringIO(text)):
                    if row.get('Start_Timestamp', '').isdigit():
                        group = str(Path(name).parent)
                        value = int(row['Start_Timestamp'])
                        csv_origins[group] = min(csv_origins.get(group, value), value)
    if not selected:
        raise ValueError('no evidence text files found')
    for name, text in selected:
        if name.endswith('.csv'):
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            identifiers = {}
            for row in rows:
                for key in ('Thread_Id', 'Process_Id', 'Address', 'Gpu_Id'):
                    if row.get(key):
                        values = identifiers.setdefault(key, {})
                        value = row[key]
                        values.setdefault(value, key.upper() + '_' + str(len(values) + 1))
                        row[key] = values[value]
                origin = csv_origins.get(str(Path(name).parent), 0)
                for key in ('Start_Timestamp', 'End_Timestamp'):
                    if row.get(key, '').isdigit(): row[key] = str(int(row[key]) - origin)
            buf = io.StringIO(newline='')
            writer = csv.DictWriter(buf, fieldnames=reader.fieldnames, lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
            text = buf.getvalue()
        text = sanitize_text(text, replacements)
        if name.endswith('.json'): json.loads(text)  # redaction must preserve valid JSON
        if PRIVATE_PATH.search(text) or EMAIL.search(text):
            raise ValueError('private identifier survived sanitization: ' + name)
        files[name] = text.encode()
    manifest = dict(schema_version=1,
        sanitization=['Known checkout/run roots replaced by variables',
                      'Other home/temp paths and emails replaced by placeholders',
                      'CSV process/device addresses replaced by stable per-file tokens',
                      'CSV timestamps shifted to a shared per-directory origin; durations preserved',
                      'No NIfTI data, binaries, source checkout or runtime shim included'],
        files=[dict(path=name, sha256=hashlib.sha256(blob).hexdigest(),
                    original_sha256=original_hashes[name]) for name, blob in sorted(files.items())])
    files['MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    files['SHA256SUMS'] = ''.join(hashlib.sha256(blob).hexdigest() + '  ' + name + '\n'
                                  for name, blob in sorted(files.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as target, gzip.GzipFile(fileobj=target, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, blob in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(blob), 0o644
                archive.addfile(member, io.BytesIO(blob))
    return hashlib.sha256(output.read_bytes()).hexdigest(), len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, metavar='LABEL=DIRECTORY')
    parser.add_argument('--redact', action='append', default=[], metavar='TOKEN=PATH')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {label: Path(path) for label, path in (value.split('=', 1) for value in args.input)}
    replacements = {path: token for token, path in (value.split('=', 1) for value in args.redact)}
    checksum, count = build_bundle(inputs, args.output, replacements)
    print(f'{checksum}  {args.output.name} ({count} sanitized evidence files)')


if __name__ == '__main__':
    main()
