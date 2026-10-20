"""Check this project's owned source archive before building outside the checkout.

This compares bytes with trusted source. It does not authenticate an archive's
publisher or make an unknown build backend safe to execute.
"""
import argparse
import configparser
from email.parser import BytesParser
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
import tomllib


FIXED_FILES = (
    'pyproject.toml', 'setup.py', 'MANIFEST.in', 'README.md',
    'requirements-dev.txt', 'requirements-lock.txt',
    'LICENSE', 'NOTICE', 'third_party/PYTORCH-LICENSE', 'csrc/pool.cpp',
    'scripts/check_installed.py', 'scripts/check_sdist.py',
    'docs/contract.md', 'docs/cpu-study.md', 'docs/results/cpu-study.json',
    'docs/exported-consumer.md', 'docs/results/export-roundtrip.json',
    'docs/linux-cpu.md',
    'docs/padded-batches.md',
    'docs/second-order.md',
    'docs/padded-export.md',
)
GENERATED_FILES = {
    'PKG-INFO', 'setup.cfg', 'src/ragged_pool.egg-info/PKG-INFO',
    'src/ragged_pool.egg-info/SOURCES.txt', 'src/ragged_pool.egg-info/dependency_links.txt',
    'src/ragged_pool.egg-info/requires.txt', 'src/ragged_pool.egg-info/top_level.txt',
}
BAD_SUFFIXES = ('.so', '.pyd', '.dll', '.dylib', '.o', '.obj', '.a', '.lib', '.exe', '.pyc', '.pyo')
BAD_DIRECTORIES = {'build', 'dist', '__pycache__', '.git', '.venv', 'venv'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def required_files(checkout):
    """Include the actual package, tests and examples, including adjacent helpers."""
    paths = set(FIXED_FILES)
    for directory in ('src/ragged_pool', 'tests', 'examples'):
        files = list((checkout / directory).rglob('*.py'))
        require(bool(files), f'The source directory is empty: {directory}')
        paths.update(path.relative_to(checkout).as_posix() for path in files)
    for name in paths:
        path = checkout / name
        require(path.is_file() and not path.is_symlink(), f'Required source is not an ordinary file: {name}')
    return paths


def check_generated_configuration(data):
    """Admit only the backend's generated egg-info settings."""
    configuration = configparser.ConfigParser(interpolation=None)
    try:
        configuration.read_string(data.decode('utf-8'))
    except (UnicodeDecodeError, configparser.Error) as error:
        raise AssertionError('Invalid generated setup.cfg') from error
    require(not configuration.defaults() and configuration.sections() == ['egg_info']
            and dict(configuration['egg_info']) == {'tag_build': '', 'tag_date': '0'},
            'Generated setup.cfg contains unexpected build configuration')


def check(checkout, sdist, extract=None):
    checkout, sdist = Path(checkout).resolve(strict=True), Path(sdist).resolve(strict=True)
    project = tomllib.loads((checkout / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    require(project['name'] == 'ragged-pool' and project['version'] == '0.1.0',
            'This checker is scoped to ragged-pool 0.1.0')
    archive_root = 'ragged_pool-0.1.0'
    required = required_files(checkout)
    payloads, seen = {}, set()
    total = 0
    # Read once: extraction uses these exact verified bytes rather than reopening
    # an archive that could have changed between checking and extraction.
    archive_bytes = sdist.read_bytes()
    require(len(archive_bytes) <= 32 * 1024**2, 'The owned source archive exceeds 32 MiB')
    import io
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as archive:
        for index, member in enumerate(archive):
            require(index < 1024, 'The owned source archive has too many members')
            name = member.name
            parts = name.split('/')
            require(name and '\\' not in name and not PurePosixPath(name).is_absolute()
                    and all(part not in ('', '.', '..') for part in parts),
                    f'Unsafe archive path: {name}')
            require(name not in seen, f'Duplicate archive member: {name}')
            seen.add(name)
            require(parts[0] == archive_root, f'Unexpected archive root: {name}')
            require(not any(part in BAD_DIRECTORIES for part in parts[1:]),
                    f'Build or runtime directory in source archive: {name}')
            require(not name.lower().endswith(BAD_SUFFIXES), f'Compiled file in source archive: {name}')
            require(member.isdir() or member.isfile(), f'Non-file archive member: {name}')
            if member.isdir():
                continue
            require(len(parts) > 1, 'The archive root must be a directory')
            relative = '/'.join(parts[1:])
            require(relative in required or relative in GENERATED_FILES,
                    f'Unexpected source archive file: {relative}')
            require(0 <= member.size <= 8 * 1024**2, f'Archive member is too large: {name}')
            total += member.size
            require(total <= 32 * 1024**2, 'Expanded source archive exceeds 32 MiB')
            stream = archive.extractfile(member)
            require(stream is not None, f'Could not read archive file: {name}')
            data = stream.read()
            require(len(data) == member.size, f'Incomplete archive file: {name}')
            if relative in required:
                require(data == (checkout / relative).read_bytes(), f'Source bytes differ: {relative}')
            payloads[relative] = data
    require(required <= payloads.keys(), 'Missing required source files: ' + ', '.join(sorted(required - payloads.keys())))
    if 'setup.cfg' in payloads:
        check_generated_configuration(payloads['setup.cfg'])
    require('PKG-INFO' in payloads, 'The source archive must contain PKG-INFO')
    metadata = BytesParser().parsebytes(payloads['PKG-INFO'])
    require(metadata['Name'] == project['name'] and metadata['Version'] == project['version'],
            'Source archive metadata differs from the project')
    require(metadata['License-Expression'] == 'MIT', 'Source archive must declare MIT')
    require(set(metadata.get_all('License-File', [])) == {'LICENSE', 'NOTICE', 'third_party/PYTORCH-LICENSE'},
            'Source archive must declare all project and PyTorch notice files')
    destination = None
    if extract is not None:
        destination = Path(extract).absolute()
        require(not destination.exists() and not destination.is_symlink(), 'Extraction directory must be new')
        require(not destination.resolve().is_relative_to(checkout), 'Extract outside the checkout')
        destination.mkdir(parents=True, exist_ok=False)
        for name, data in sorted(payloads.items()):
            path = destination.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
    return {'status': 'passed', 'sdist_sha256': sha256(archive_bytes).hexdigest(),
            'archive_root': archive_root, 'size_bytes': len(archive_bytes),
            'expanded_size_bytes': total, 'files': sorted(payloads),
            'required_file_sha256': {name: sha256(payloads[name]).hexdigest() for name in sorted(required)},
            'extracted_to': str(destination) if destination is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--sdist', type=Path, required=True)
    parser.add_argument('--extract', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Receipt path must be new')
    result = check(args.checkout, args.sdist, args.extract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'sdist_sha256': result['sdist_sha256']}, indent=2))


if __name__ == '__main__':
    main()
