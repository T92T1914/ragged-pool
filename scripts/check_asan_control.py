"""Qualify the sanitizer toolchain with a separate, intentionally invalid binary."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    source = args.directory / "control.cpp"
    binary = args.directory / "control.so"
    source.write_text(
        'extern "C" void trigger() { int* p = new int[4]; volatile int index = 8; '
        "p[index] = 1; delete[] p; }\n"
        'extern "C" int throw_and_catch() { try { throw 7; } catch (int value) { return value; } }\n',
        encoding="utf-8",
    )
    subprocess.run([os.environ["CXX"], "-std=c++17", "-O0", "-g", "-fPIC", "-shared", "-fsanitize=address",
                    str(source), "-o", str(binary)], check=True)
    environment = dict(os.environ, LD_PRELOAD=f'{os.environ["ASAN_RUNTIME"]}:{os.environ["CXX_STDLIB"]}')
    exception = subprocess.run(
        [os.sys.executable, "-I", "-c",
         "import ctypes,sys; print(ctypes.CDLL(sys.argv[1]).throw_and_catch())", str(binary)],
        env=environment, capture_output=True, text=True,
    )
    (args.directory / "exception.stdout").write_text(exception.stdout, encoding="utf-8")
    (args.directory / "exception.stderr").write_text(exception.stderr, encoding="utf-8")
    if exception.returncode != 0 or exception.stdout.strip() != "7":
        raise RuntimeError(f"Expected C++ exception control failed: {exception.returncode}")
    result = subprocess.run([os.sys.executable, "-I", "-c",
                             "import ctypes,sys; ctypes.CDLL(sys.argv[1]).trigger()", str(binary)],
                            env=environment, capture_output=True, text=True)
    (args.directory / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode == 0 or "AddressSanitizer: heap-buffer-overflow" not in result.stderr:
        raise RuntimeError(f"Expected injected ASan failure was not observed: {result.returncode}")
    print("C++ exception control passed; injected heap-buffer-overflow detected. These qualify the tool, not the operator.")


if __name__ == "__main__":
    main()
