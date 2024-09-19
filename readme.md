# Prolate_swf

## Note

This Python module wraps directly the `prolate_swf.f90` code at <https://github.com/MathieuandSpheroidalWaveFunctions/prolate_swf>, providing access to the subroutine
version of the prolate spheroidal wave function in that file.

Once installed in Python, the package can be used thus:

    from spheroidalwavefunctions import prolate_swf
    r = prolate_swf.profcn(c=0.5, m=0, lnum=10, x1=0.5, ioprad=2, iopang=2, iopnorm=0, arg=[0.1, 0.2])

Function calling details can be viewed via:

    print(prolate_swf.profcn.__doc__)

and further details on the function parameters are available [below](#input-and-output). A wrapper that provides a similar interface to
the spheroidal functions in scipy is available in the Utilities module of
[echoSMs](https://github.com/ices-tools-dev/echoSMs).

The original readme file, modified to work better in Markdown, is available [here](readme_original.md).

