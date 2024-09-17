"""Prolate spheroidal wave functions."""

from collections import namedtuple
from spheroidalwavefunctions import prolate_swf
import numpy as np
import os
os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin')

swf_t = namedtuple('swf', ['r1c', 'ir1e', 'r1dc', 'ir1de', 'r2c', 'ir2e', 'r2dc', 'ir2de',
                           'naccr', 's1c', 'is1e', 's1dc', 'is1de', 'naccs'])


def pro_ang1(m, n, c, x):
    """Prolate spheroidal angular function of the first kind and derivative.

    Parameters
    ----------
    m : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    a = prolate_swf.profcn(c=c, m=m, lnum=n-m+1, x1=0.0, ioprad=0, iopang=2, iopnorm=0, arg=[x])
    p = swf_t._make(a)
    s = p.s1c * np.float_power(10.0, p.is1e)
    sp = p.s1dc * np.float_power(10.0, p.is1de)

    return s[n][0], sp[n][0]


def pro_rad1(m, n, c, x):
    """Prolate spheroidal radial function of the first kind and derivative.

    Parameters
    ----------
    m : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    a = prolate_swf.profcn(c=c, m=m, lnum=n-m+1, x1=x-1.0, ioprad=1, iopang=0, iopnorm=0, arg=[0])
    p = swf_t._make(a)
    s = p.r1c * np.float_power(10.0, p.ir1e)
    sp = p.r1dc * np.float_power(10.0, p.ir1de)

    return s[n][0], sp[n][0]


def pro_rad2(m, n, c, x):
    """Prolate spheroidal radial function of the second kind and derivative.

    Parameters
    ----------
    m : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    ioprad = 1 if x-1.0 < 1e-10 else 2
    a = prolate_swf.profcn(c=c, m=m, lnum=n-m+1, x1=x-1.0,
                           ioprad=ioprad, iopang=0, iopnorm=0, arg=[0])
    p = swf_t._make(a)
    if ioprad == 1:
        s = np.nan
        sp = np.nan
    else:
        s = p.r2c * np.float_power(10.0, p.ir2e)
        sp = p.r2dc * np.float_power(10.0, p.ir2de)

    return s[n][0], sp[n][0]
