import pytest
from math import isclose

def test_swf_import():
    from spheroidalwavefunctions import prolate_swf

def test_swf_calculate():
    from spheroidalwavefunctions import prolate_swf

    # Very simple check that at least some of the output is correct
    r = prolate_swf.profcn(c=0.5, m=0, lnum=10, x1=0.5, ioprad=2, iopang=2, iopnorm=0, arg=[0.1, 0.2])
    assert isclose(r[0][0], 9.355129525869243)
