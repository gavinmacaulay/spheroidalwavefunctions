# Prolate spheroidal wave functions

![Static badge](https://img.shields.io/pypi/v/spheroidalwavefunctions.svg)
![Static badge](https://img.shields.io/github/license/gavinmacaulay/spheroidalwavefunctions)
[![python](https://img.shields.io/pypi/pyversions/spheroidalwavefunctions.svg?logo=python&logoColor=white)](https://pypi.org/project/https://github.com/gavinmacaulay/spheroidalwavefunctions/)

A Python module that wraps directly the `prolate_swf.f90` code at <https://github.com/MathieuandSpheroidalWaveFunctions/prolate_swf>, providing access to the subroutine
version of the prolate spheroidal wave code in that file.

Once installed in Python, the package can be used thus:

    from spheroidalwavefunctions import prolate_swf
    r = prolate_swf.profcn(c=0.5, m=0, lnum=10, x1=0.5, ioprad=2, iopang=2, iopnorm=0, arg=[0.1, 0.2])

Function calling details can be viewed via:

    print(prolate_swf.profcn.__doc__)

and further details on the function parameters are available in the [documentation](https://github.com/gavinmacaulay/spheroidalwavefunctions/blob/master/readme_original.md). A wrapper that provides a similar interface to
the spheroidal functions in scipy is available in the Utilities module of
[echoSMs](https://github.com/ices-tools-dev/echoSMs).

The original readme file, modified to work better in Markdown, is available [here](https://github.com/gavinmacaulay/spheroidalwavefunctions/blob/master/readme_original.md).

## Building

Building the Python package is most easily done on Linux with C, C++ and Fortran compilers installed.

It is easiest to leave the building of macOS and Windows packages to the `cibuildwheel` Github action that runs on tagged commits.

The Fortran code can be compiled (after installing `uv`) with:

```bash
uv tool install --with numpy meson
uvx meson setup tt
uvx meson compile -C tt
```

(or install `meson` directly and don't use the uv/uvx commands).

This will generate a .dll or .so file in the `tt` directory.

More usefully, a Python package can be created with:

```bash
uv build
uvx repairwheel -o whl dist/*.whl
```

This will create Python .whl files in the `whl` directory.
  