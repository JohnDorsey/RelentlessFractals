import math

import TestingBasics
from TestingBasics import assert_equal, assert_less, assert_nearly_equal

FUN_REPLACEMENT_ERROR_TOLERANCE = 10.0**-12
_assertNearlyEqualTrig = TestingBasics.non_overridably_curry_kwarg_dict(assert_nearly_equal, {"error_tolerance":FUN_REPLACEMENT_ERROR_TOLERANCE})


atan = math.atan
lousy_sin = math.sin
lousy_cos = math.cos

pi = math.pi
tau = 2.0*pi
half_pi = pi/2.0


def sin(value):
    if isinstance(value, complex):
        return complex(sin(value.real)*math.cosh(value.imag), cos(value.real)*math.sinh(value.imag))
    else:
        fullSubValue = value % tau
        halfSubValue = fullSubValue % pi
        quadSubValue = min(halfSubValue, pi-halfSubValue)
        return lousy_sin(quadSubValue) * (-1 if fullSubValue > pi else 1)
    
def cos(value):
    # https://ximera.osu.edu/math/complexBook/complexBook/complexTrig/complexTrig
    if isinstance(value, complex):
        return complex(cos(value.real)*math.cosh(value.imag), -sin(value.real)*math.sinh(value.imag))
    else:
        fullSubValue = value % tau
        halfSubValue = min(fullSubValue, tau-fullSubValue)
        quadSubValue = min(halfSubValue, pi-halfSubValue)
        return lousy_cos(quadSubValue) * (-1 if halfSubValue > half_pi else 1)
    
def tan(value):
    if isinstance(value, complex):
        raise NotImplementedError()
        # https://www.redcrab-software.com/en/Calculator/Algebra/Complex/Tan
    else:
        sinVal, cosVal = sin(value), cos(value)
        try:
            return sinVal/cosVal
        except ZeroDivisionError:
            return math.inf
    
for i in range(-1000,1000,1):
    v = i / 100.0
    assert_less(abs(sin(v)-lousy_sin(v)), FUN_REPLACEMENT_ERROR_TOLERANCE)
    assert_less(abs(cos(v)-lousy_cos(v)), FUN_REPLACEMENT_ERROR_TOLERANCE)
    assert_less(abs(sin(i*pi)), FUN_REPLACEMENT_ERROR_TOLERANCE)
    _assertNearlyEqualTrig(sin((i+0.25)*tau), 1)
    _assertNearlyEqualTrig(sin((i+0.75)*tau), -1)
    _assertNearlyEqualTrig(cos((i+0.5)*pi), 0)
    _assertNearlyEqualTrig(cos(i*tau), 1)
    _assertNearlyEqualTrig(cos((i+0.5)*tau), -1)
    
    for fun, arg, expected in [
            (sin, 1+1j, 1.2984575814159773 + 0.6349639147847361j),
            (sin,1-2j,3.165778513216168 - 1.9596010414216063j),
            (cos,1-2j,2.0327230070196656 + 3.0518977991518j),
            (sin,1+10j, 9267.31595119809 + 5950.475117265045j),
            (cos, 1+10j, 5950.475141794733 - 9267.315912995366j)]:
        _assertNearlyEqualTrig(fun(arg), expected), (fun, arg, expected)



"""
def arccos(value):
    # https://www.hpmuseum.org/forum/thread-410-post-2672.html#pid2672
    if isinstance(value, complex)
"""
    
    
    
    
    