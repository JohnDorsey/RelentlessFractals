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
    fullSubValue = value % tau
    halfSubValue = fullSubValue % pi
    quadSubValue = min(halfSubValue, pi-halfSubValue)
    return lousy_sin(quadSubValue) * (-1 if fullSubValue > pi else 1)
    
def cos(value):
    fullSubValue = value % tau
    halfSubValue = min(fullSubValue, tau-fullSubValue)
    quadSubValue = min(halfSubValue, pi-halfSubValue)
    return lousy_cos(quadSubValue) * (-1 if halfSubValue > half_pi else 1)
    
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

    
    
    
    
    