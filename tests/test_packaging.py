"""Check source and installed artifact identity without compiling or importing Torch."""
from hashlib import sha256
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


def load_checker(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    specification = importlib.util.spec_from_file_location('packaging_test_' + name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


sdist_checker = load_checker('check_sdist')
installed_checker = load_checker('check_installed')
METADATA = (b'Metadata-Version: 2.4\nName: ragged-pool\nVersion: 0.1.0\n'
            b'License-Expression: MIT\nLicense-File: LICENSE\nLicense-File: NOTICE\n'
            b'License-File: third_party/PYTORCH-LICENSE\n\n')


@pytest.fixture
def source_fixture(tmp_path):
    root = tmp_path / 'trusted-source'
    files = set(sdist_checker.FIXED_FILES) | {
        'src/ragged_pool/__init__.py', 'tests/conftest.py', 'tests/generated_cases.py',
        'tests/test_generated.py', 'examples/learn_sets.py',
    }
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('source bytes for ' + name + '\r\n').encode())
    (root / 'pyproject.toml').write_text('[project]\nname="ragged-pool"\nversion="0.1.0"\n', encoding='utf-8')
    return root


def source_members(root):
    members = []
    for name in sorted(sdist_checker.required_files(root)):
        info = tarfile.TarInfo('ragged_pool-0.1.0/' + name)
        data = (root / name).read_bytes()
        info.size = len(data)
        members.append((info, data))
    info = tarfile.TarInfo('ragged_pool-0.1.0/setup.cfg')
    data = b'[egg_info]\ntag_build = \ntag_date = 0\n\n'
    info.size = len(data)
    members.append((info, data))
    info = tarfile.TarInfo('ragged_pool-0.1.0/PKG-INFO')
    info.size = len(METADATA)
    members.append((info, METADATA))
    return members


def write_archive(path, members):
    with tarfile.open(path, 'w:gz') as archive:
        for info, data in members:
            archive.addfile(info, BytesIO(data) if info.isfile() else None)


def test_sdist_contains_adjacent_helpers_and_extracts_verified_bytes(source_fixture, tmp_path):
    archive, destination = tmp_path / 'owned.tar.gz', tmp_path / 'extracted'
    write_archive(archive, source_members(source_fixture))
    receipt = sdist_checker.check(source_fixture, archive, destination)
    assert receipt['status'] == 'passed'
    assert receipt['sdist_sha256'] == sha256(archive.read_bytes()).hexdigest()
    for name in ('tests/generated_cases.py', 'tests/conftest.py', 'scripts/check_installed.py'):
        expected = (source_fixture / name).read_bytes()
        assert (destination / name).read_bytes() == expected
        assert receipt['required_file_sha256'][name] == sha256(expected).hexdigest()
    assert not (destination / 'ragged_pool-0.1.0').exists()


@pytest.mark.parametrize('fault', ['missing_helper', 'changed_source', 'duplicate', 'absolute',
                                  'traversal', 'backslash', 'root', 'symlink', 'hardlink',
                                  'fifo', 'binary', 'build_directory', 'unknown', 'metadata'])
def test_bad_archive_is_rejected_before_any_extraction(source_fixture, tmp_path, fault):
    members = source_members(source_fixture)
    if fault == 'missing_helper':
        members = [(info, data) for info, data in members if not info.name.endswith('/generated_cases.py')]
    elif fault == 'changed_source':
        info, data = members[0]
        replacement = tarfile.TarInfo(info.name)
        replacement.size = len(data) + 1
        members[0] = replacement, data + b'x'
    elif fault == 'duplicate':
        members.append(members[0])
    elif fault == 'metadata':
        info = tarfile.TarInfo('ragged_pool-0.1.0/PKG-INFO')
        data = METADATA.replace(b'License-Expression: MIT', b'License-Expression: BSD-3-Clause')
        info.size = len(data)
        members[-1] = info, data
    else:
        name = {
            'absolute': '/outside.py', 'traversal': 'ragged_pool-0.1.0/../outside.py',
            'backslash': 'ragged_pool-0.1.0/..\\outside.py', 'root': 'wrong-0.1.0/source.py',
            'binary': 'ragged_pool-0.1.0/src/ragged_pool/_C.so',
            'build_directory': 'ragged_pool-0.1.0/build/stale.txt',
            'unknown': 'ragged_pool-0.1.0/unreviewed.py',
        }.get(fault, 'ragged_pool-0.1.0/link')
        info = tarfile.TarInfo(name)
        if fault in ('symlink', 'hardlink', 'fifo'):
            info.type = {'symlink': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE, 'fifo': tarfile.FIFOTYPE}[fault]
            info.linkname = '../outside'
        members.append((info, b''))
    archive, destination = tmp_path / 'bad.tar.gz', tmp_path / 'must-not-exist'
    write_archive(archive, members)
    with pytest.raises(AssertionError):
        sdist_checker.check(source_fixture, archive, destination)
    assert not destination.exists()


def test_generated_configuration_cannot_add_unverified_build_settings(source_fixture, tmp_path):
    members = source_members(source_fixture)
    for index, (info, data) in enumerate(members):
        if info.name.endswith('/setup.cfg'):
            replacement = tarfile.TarInfo(info.name)
            altered = data + b'[build_ext]\ninclude_dirs = /unverified/headers\n'
            replacement.size = len(altered)
            members[index] = replacement, altered
    archive, destination = tmp_path / 'altered-config.tar.gz', tmp_path / 'must-not-exist'
    write_archive(archive, members)
    with pytest.raises(AssertionError, match='unexpected build configuration'):
        sdist_checker.check(source_fixture, archive, destination)
    assert not destination.exists()


@pytest.mark.parametrize(('names', 'suffixes', 'accepted'), [
    (['ragged_pool/_C.cp311-win_amd64.pyd'], ['.cp311-win_amd64.pyd', '.pyd'], True),
    (['ragged_pool/_C.cpython-311-x86_64-linux-gnu.so'], ['.cpython-311-x86_64-linux-gnu.so', '.so'], True),
    (['ragged_pool/_C.cp311-win_amd64.pyd'], ['.cpython-311-x86_64-linux-gnu.so', '.so'], False),
    (['ragged_pool/_C.so', 'ragged_pool/_C.old.so'], ['.so'], False),
    (['ragged_pool/_Cunrelated.so'], ['.so'], False),
    ([], ['.so'], False),
])
def test_native_selection_uses_interpreter_suffix_and_rejects_ambiguity(names, suffixes, accepted):
    if accepted:
        assert installed_checker.native_entry(names, suffixes) == names[0]
    else:
        with pytest.raises(AssertionError, match='exactly one native library'):
            installed_checker.native_entry(names, suffixes)


@pytest.fixture
def installed_fixture(tmp_path, monkeypatch):
    checkout, prefix, consumer = tmp_path / 'checkout', tmp_path / 'acceptance', tmp_path / 'consumer'
    consumer.mkdir()
    monkeypatch.chdir(consumer)
    monkeypatch.setattr(sys, 'prefix', str(prefix))
    monkeypatch.setattr(sys, 'version_info', (3, 11, 8))
    native = 'ragged_pool/_C' + installed_checker.EXTENSION_SUFFIXES[0]
    files = {'ragged_pool/__init__.py': b'# original package\n', native: b'owned native bytes'}
    for name in ('LICENSE', 'NOTICE', 'third_party/PYTORCH-LICENSE'):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('notice ' + name).encode())
        files['ragged_pool-0.1.0.dist-info/licenses/' + name] = path.read_bytes()
    source = checkout / 'src/ragged_pool/__init__.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(files['ragged_pool/__init__.py'])
    files['ragged_pool-0.1.0.dist-info/METADATA'] = METADATA
    for name, data in files.items():
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    wheel = tmp_path / 'owned.whl'
    with ZipFile(wheel, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    distribution = SimpleNamespace(version='0.1.0', locate_file=lambda name: prefix / name)
    monkeypatch.setattr(installed_checker, 'distribution', lambda name: distribution)
    package = SimpleNamespace(__file__=str(prefix / 'ragged_pool/__init__.py'))
    torch = SimpleNamespace(__version__='2.9.1+cpu',
                            ops=SimpleNamespace(loaded_libraries={str(prefix / native)}))
    monkeypatch.setitem(sys.modules, 'ragged_pool', package)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    return SimpleNamespace(checkout=checkout, prefix=prefix, wheel=wheel, native=native,
                           package=package, torch=torch)


def test_installed_check_identifies_exact_wheel_and_loaded_library(installed_fixture):
    fixture = installed_fixture
    receipt = installed_checker.check(fixture.checkout, fixture.wheel)
    assert receipt['native_library'] == fixture.native
    assert receipt['native_library_loaded'] is True
    assert receipt['package_file_sha256'][fixture.native] == sha256(b'owned native bytes').hexdigest()


@pytest.mark.parametrize('fault', ['native_bytes', 'source_bytes', 'import_path', 'loaded_path', 'duplicate'])
def test_installed_identity_rejects_substitution(installed_fixture, fault):
    fixture = installed_fixture
    if fault == 'native_bytes':
        (fixture.prefix / fixture.native).write_bytes(b'different library')
    elif fault == 'source_bytes':
        (fixture.checkout / 'src/ragged_pool/__init__.py').write_bytes(b'# changed source\n')
    elif fault == 'import_path':
        fixture.package.__file__ = str(fixture.checkout / 'src/ragged_pool/__init__.py')
    elif fault == 'loaded_path':
        fixture.torch.ops.loaded_libraries = {str(fixture.prefix / 'unverified.so')}
    else:
        with pytest.warns(UserWarning, match='Duplicate name'):
            with ZipFile(fixture.wheel, 'a') as archive:
                archive.writestr(fixture.native, b'owned native bytes')
    with pytest.raises(AssertionError):
        installed_checker.check(fixture.checkout, fixture.wheel)
