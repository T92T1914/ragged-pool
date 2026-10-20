"""Check that an installed wheel matches its checkout and carries its licenses."""
import argparse
from email.parser import BytesParser
from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import distribution
import json
from pathlib import Path
import sys
from zipfile import ZipFile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def native_entry(names, suffixes=EXTENSION_SUFFIXES):
    """Select the one native module this interpreter can load."""
    candidates = [name for name in names
                  if name.startswith('ragged_pool/_C') and name.endswith(('.so', '.pyd'))]
    supported = {'ragged_pool/_C' + suffix for suffix in suffixes}
    require(len(candidates) == 1 and candidates[0] in supported,
            'The wheel must contain exactly one native library for this interpreter')
    return candidates[0]


def check(checkout, wheel):
    checkout = checkout.resolve(strict=True)
    wheel = wheel.resolve(strict=True)
    require(not Path.cwd().resolve().is_relative_to(checkout), 'Run from outside the checkout')
    require(not Path(sys.prefix).resolve().is_relative_to(checkout), 'Use an environment outside the checkout')
    installed = distribution('ragged-pool')
    source = {
        path.relative_to(checkout / 'src').as_posix(): path.read_bytes()
        for path in (checkout / 'src' / 'ragged_pool').rglob('*.py')
    }
    require(bool(source), 'The checkout must contain the Python package source')
    hashes = {}
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'The wheel must not contain duplicate members')
        metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
        require(len(metadata_names) == 1, 'The wheel must contain one distribution metadata record')
        metadata_name = metadata_names[0]
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        require(metadata['Name'] == 'ragged-pool', 'Unexpected wheel distribution')
        require(metadata['Version'] == installed.version, 'Installed and wheel versions differ')
        require(metadata['License-Expression'] == 'MIT', 'The wheel must declare the MIT license')
        wheel_python = {name for name in names if name.startswith('ragged_pool/') and name.endswith('.py')}
        require(wheel_python == source.keys(), 'Wheel and checkout Python source sets differ')
        native = native_entry(names)
        for name in sorted(wheel_python | {native}):
            wheel_bytes = archive.read(name)
            installed_path = Path(installed.locate_file(name)).resolve(strict=True)
            require(not installed_path.is_relative_to(checkout), f'Package source leaked into acceptance: {name}')
            require(installed_path.is_relative_to(Path(sys.prefix).resolve()), f'Package is outside the acceptance environment: {name}')
            require(installed_path.read_bytes() == wheel_bytes, f'Installed bytes differ from the wheel: {name}')
            if name in source:
                require(source[name] == wheel_bytes, f'Wheel bytes differ from the checkout: {name}')
            hashes[name] = sha256(wheel_bytes).hexdigest()
        dist_info = metadata_name.rsplit('/', 1)[0]
        license_files = ('LICENSE', 'NOTICE', 'third_party/PYTORCH-LICENSE')
        require(set(metadata.get_all('License-File', [])) == set(license_files),
                'The wheel must declare all project and PyTorch notice files')
        for name in license_files:
            entry = f'{dist_info}/licenses/{name}'
            expected = (checkout / name).read_bytes()
            require(entry in names, f'The wheel is missing {name}')
            require(archive.read(entry) == expected, f'Wheel {name} differs from the checkout')
            require(Path(installed.locate_file(entry)).read_bytes() == expected, f'Installed {name} differs from the checkout')

    import ragged_pool
    import torch

    require(Path(ragged_pool.__file__).resolve() == Path(installed.locate_file('ragged_pool/__init__.py')).resolve(),
            'The imported package differs from the verified installation')
    native_path = Path(installed.locate_file(native)).resolve(strict=True)
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
    require(native_path in loaded, 'The verified native library was not loaded by PyTorch')
    require(torch.__version__ == '2.9.1+cpu', 'Acceptance requires PyTorch 2.9.1+cpu')
    require(sys.version_info[:3] == (3, 11, 8), 'Acceptance requires CPython 3.11.8')
    return {'package_version': installed.version, 'torch_version': torch.__version__,
            'python_version': sys.version.split()[0], 'license_expression': metadata['License-Expression'],
            'native_library': native, 'native_library_loaded': True,
            'package_file_sha256': hashes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.checkout, args.wheel), indent=2))
