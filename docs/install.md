# Install the v0.2.0 CPU release

The [v0.2.0 release](https://github.com/T92T1914/ragged-pool/releases/tag/v0.2.0) supplies the Windows x64 CPU wheel and a source archive from the same release revision. Download these three assets into a new directory, such as `C:\work\ragged-0.2.0\downloads`:

- `ragged_pool-0.2.0-cp311-cp311-win_amd64.whl` contains the Python package and native CPU extension.
- `ragged_pool-source-<full-release-commit>.zip` contains the tracked source, examples, guides and license notices. Its actual filename contains the full release commit. This is a Git source archive, not a Python source distribution.
- `SHA256SUMS.txt` identifies the wheel and source-archive bytes. Compare it with `Get-FileHash -Algorithm SHA256` before using the files. A matching hash identifies bytes and does not authenticate an unknown publisher.

Use Windows x64 and CPython 3.11.8. The wheel requires exactly `torch==2.9.1+cpu`, installed separately from the official PyTorch index. It needs no compiler at installation. Other Python, PyTorch and platform combinations have not been qualified by this wheel.

## Run the installed example

These PowerShell commands use new environment, source and consumer directories outside any checkout. Keep only the two selected release payloads and checksum file in the download directory. The source archive supplies the pinned requirements and example, while imports come from the installed wheel.

```powershell
$root = 'C:\work\ragged-0.2.0'
$downloads = Join-Path $root 'downloads'
$wheel = Join-Path $downloads 'ragged_pool-0.2.0-cp311-cp311-win_amd64.whl'
$archives = @(Get-ChildItem -LiteralPath $downloads -Filter 'ragged_pool-source-*.zip' -File)
if ($archives.Count -ne 1) { throw 'Select exactly the source archive from this release' }
Get-FileHash -LiteralPath $wheel,$archives[0].FullName -Algorithm SHA256
Get-Content -LiteralPath (Join-Path $downloads 'SHA256SUMS.txt')
# Compare both filename/hash pairs before continuing.
$source = Join-Path $root 'source'
if (Test-Path -LiteralPath $source) { throw 'Use a new source directory' }
Expand-Archive -LiteralPath $archives[0].FullName -DestinationPath $source
if (Test-Path -LiteralPath (Join-Path $root 'installed-env')) { throw 'Use a new installation environment' }
py -3.11 -m venv (Join-Path $root 'installed-env')
$python = Join-Path $root 'installed-env\Scripts\python.exe'
& $python -I -c "import sys; assert sys.version_info[:3] == (3, 11, 8), sys.version"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = ''
& $python -m pip --isolated install 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu
& $python -m pip --isolated install -r (Join-Path $source 'requirements-lock.txt')
& $python -m pip --isolated install --no-deps $wheel
& $python -m pip check
$consumer = Join-Path $root 'consumer'
New-Item -ItemType Directory -Path $consumer | Out-Null
Copy-Item -LiteralPath (Join-Path $source 'examples\train_sets.py') -Destination $consumer
Set-Location -LiteralPath $consumer
& $python -I .\train_sets.py --dtype float32
& $python -I .\train_sets.py --dtype float64
```

Stop if a command fails. Each successful example checks native/reference outputs, loss, input and parameter gradients, then performs one SGD step and compares updated parameters. The JSON result reports the installed module path, package version, dtype and actual optimizer-step count. The package path should be under `installed-env\Lib\site-packages`, outside the extracted source. This is synthetic training integration, with no accuracy or performance claim.

The README's [two-item pooling example](../README.md#ragged-pool) also works with this installation. The empty second group returns a zero vector, and `backward()` sends gradients to values and logits.

## Source builds and other capabilities

The [Windows build recipe](../README.md#build-and-run) uses an existing MSVC 14.44 x64 toolchain and C++17. The separate [Linux source-to-wheel recipe](linux-cpu.md) qualifies Ubuntu 24.04 x86-64, GCC 13, CPython 3.11.8 and PyTorch 2.9.1+cpu. A wheel named `linux_x86_64` is not a manylinux compatibility claim.

The default CPU wheel contains no CUDA extension and requires the CPU PyTorch build. The [CUDA companion](examples.md#cuda-companion) and [historical co-installation route](coinstallation.md) identify their own accepted 0.1.0 packages, framework variant, Windows hardware and diagnostic limits. Those older package identities remain part of those routes. This release does not qualify a new CUDA wheel or a new mixed CPU/CUDA pair.

The source and wheel retain MIT, NOTICE and the full retained PyTorch license. PyTorch dependency binaries are not bundled. See the [release notes](releases/v0.2.0.md) for the current snapshot and the [contract](contract.md) for supported tensors, derivatives and exclusions. Trusted model artifacts keep exact environment and native-byte identities. A newly installed release does not automatically accept an artifact produced by another binary.
