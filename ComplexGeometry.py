

from enum import Enum
import itertools


import TestingBasics
from SeqTests import get_shared_value
from TestingBasics import test_nearly_equal, assert_nearly_equal, raises_instanceof, assert_equal, assert_isinstance, AssuranceError

import Trig
from math import inf as infinity


assert Trig.sin(Trig.tau) == 0.0
assert Trig.cos(Trig.tau) == 1.0





class SpecialAnswer(Enum):
    # FORMALLY_UNDEFINED = "formally_undefined"
    DNE = "dne"
    ERROR = "error"
    ZERO_MOD_TAU = "zero_mod_tau"
    VERTICAL_SLOPE = "vertical_slope"
    ORIGIN = "origin"

    
class ComplexOnPolarSeam:
    def __init__(self, real, imag):
        if imag != 0.0:
            raise ValueError("not on seam.")
        if real == 0.0:
            raise ValueError("this is the origin! is the origin part of the seam?")
        # super().__init__(real, imag)
        self.real, self.imag = (real, imag)
    def __abs__(self):
        assert self.imag == 0.0
        assert self.real > 0.0
        return self.real
    def __repr__(self):
        return "ComplexOnPolarSeam({}, {})".format(self.real, self.imag)
    def __eq__(self, other):
        assert self.imag == 0
        if isinstance(other, (complex, ComplexOnPolarSeam)):
            assert other.imag == 0
            return ((self.real == other.real) and (self.imag == other.imag))
        else:
            return False







def real_of(val):
    return val.real
    
def imag_of(val):
    return val.imag
    
def inv_abs_of(val):
    if val == 0:
        return infinity
    return 1.0/abs(val)




"""
def peek_multi_as_tuple_and_iter(input_seq, count=None):
    assert count is not None
    inputGen = iter(input_seq)
    startingItems = tuple(itertools.islice(inputGen, None, count))
    assert len(startingItems) <= count
    assert len(startingItems) == count, "not enough items!"
    return (startingItems, inputGen)
"""


"""
def trisign(value): # copied from /Geode/photo.py
    return compare(0, value)
        
        
def compare(a, b): # copied from /Geode/photo.py
    if a < b:
        return 1
    elif a > b:
        return -1
    else:
        assert a == b
        return 0
"""


"""
def gen_differences(input_seq):
    for pair in gen_track_previous_full(input_seq):
        yield pair[1] - pair[0]
        


def contains_nonzero_opposite_signs(input_seq):
    containsNegatives, containsPositives = (False, False)
    for item in input_seq:
        if item > 0:
            if containsNegatives == True:
                return True
            containsPositives = True
        if item < 0:
            if containsPositives == True:
                return True
            containsNegatives = True
    return False
    
assert not contains_nonzero_opposite_signs([5, 7, 3, 4])
assert not contains_nonzero_opposite_signs([5, 0.0, -0.0, 4, 0, 4.3, -0])
assert not contains_nonzero_opposite_signs([-5, 0.0, -0.0, -4, 0, -4.3, -0])
assert contains_nonzero_opposite_signs([-5, 0.0, -0.0, 4, 0, -4.3, -0])
    

def is_weakly_monotonic(input_seq): # strongly would not allow horiz line test fail.
    return contains_nonzero_opposite_signs(gen_differences(input_seq))
    
assert is_weakly_monotonic([5.5, 6.6, 6.6, 7.7])
assert is_weakly_monotonic([5.5, 4.4, 4.4, 3.3])
assert not is_weakly_monotonic([5.5, 6.6, 6.5999, 7.7])
assert not is_weakly_monotonic([5.5, 4.398, 4.399, 3.3])
"""


"""
def float_composition_magnitude(val):
    return 0 if val == 0 else 1
    
def float_composition_sign(val):
    sign = -1 if val < 0.0 else 1 if val > 0.0 else None
    if sign is None:
        sign = -1 if str(val)[0] == "-" else 1
    return sign
    
def float_composition_positive(val):
    return float_composition_sign(val) > 0
"""

"""
def get_complex_angle(c):
    if c.real == 0:
        c = c + ZERO_DIVISION_NUDGE
    return math.atan(c.imag/c.real) + (math.pi if c.real < 0 else (2*math.pi if c.imag <= 0 else 0.0))
"""
"""
            # return UndefinedAtBoundary
            if extra_assertions:
                otherResult = "DEFAULT"
                try:
                    cconj = c.conjugate()
                    otherResult = get_complex_angle(cconj, extra_assertions=False)
                except UndefinedAtBoundaryError as uabe:
                    raise UndefinedAtBoundaryError("can't get complex angle of point on seam, with extra assertions.")
                assert False, "asymmetrical failure for c={}, cconj={}, (c==cconj)={}, otherResult={}. ".format(c, cconj, (c==cconj), otherResult)
            else:
                raise UndefinedAtBoundaryError("can't get complex angle of point on seam, without extra assertions.")

"""
"""
def get_complex_angle(c):
    if c.imag == 0:
        if c.real > 0:
            return 0.0
        elif c.real < 0:
            return math.pi
        else:
            assert c.real == 0
            
            return 0.0
                
    if c.real == 0:
        return (0.5*math.pi if c.imag >= 0 else 1.5*math.pi)
    if c.imag < 0:
        return math.pi + get_complex_angle(complex(-c.real, assure_positive(-c.imag)))
    if c.real < 0:
        # return math.pi*0.5 + get_complex_angle(complex(c.imag, assure_positive(-c.real)
        return math.pi*0.5 + get_complex_angle(c/complex(0,1))
    return math.atan(c.imag/c.real)
"""
"""
def get_complex_angle(c):
    if c.imag == 0:
        if c.real == 0:
            raise UndefinedAtBoundaryError("origin has no angle!")
        else:
            if float_composition_positive(c.real):
                return 0.0
            else:
                return math.pi
    else:
        if c.real == 0:
            assert c.imag != 0
            return (0.5*math.pi if float_composition_positive(c.imag) else 1.5*math.pi)
        else:
            if not float_composition_positive(c.imag):
                return math.pi + get_complex_angle(complex(-c.real, assure_positive(-c.imag)))
            if not float_composition_positive(c.real):
                # return math.pi*0.5 + get_complex_angle(complex(c.imag, assure_positive(-c.real)
                return math.pi*0.5 + get_complex_angle(div_complex_by_i(c))
            return math.atan(c.imag/c.real)
    
assert_nearly_equal(get_complex_angle(2+2j), math.pi/4.0)
assert_nearly_equal(get_complex_angle(-2+2j), 3*math.pi/4.0)
assert_nearly_equal(get_complex_angle(-2-2j), 5*math.pi/4.0)
assert_nearly_equal(get_complex_angle(2-2j), 7*math.pi/4.0)

assert_nearly_equal(get_complex_angle(1j), math.pi/2.0)
assert_nearly_equal(get_complex_angle(-1j), 1.5*math.pi)
"""




















def float_range(start, stop, step):
    assert stop > start
    assert step > 0
    current = start
    for i in itertools.count(1):
        assert start <= current < stop
        yield current
        current = start + step*i
        if current >= stop:
            break
assert_equal(list(float_range(2,3,0.2)), [2.0, 2.2, 2.4, 2.6, 2.8])
assert list(float_range(2,3.0000001,0.2)) == [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]






"""
def get_normalized(value):
    if value == 0:
        assert isinstance(value, complex)
        # return get_normalized(complex(math.copysign(1,value.real), math.copysign(1,value.imag)))
        print("get_normalized will return its default value. This shouldn't happen often.")
        return complex(1,0)
    return value / abs(value)
# assert get_normalized(complex(0.0,-0.0)) = complex(0.0,-1.0)
assert_nearly_equal(get_normalized(complex(3,3)), complex(2**0.5/2, 2**0.5/2))
"""

def get_normalized(value, undefined_result=SpecialAnswer.DNE):
    if value == 0:
        assert isinstance(value, complex)
        return undefined_result
    return value / abs(value)
    
lastResult = None
for theta in float_range(-0.1, Trig.tau+0.1, 0.05):
    result = get_shared_value((get_normalized(complex(Trig.cos(theta)*s, Trig.sin(theta)*s)) for s in [0.01, 0.1, 1, 10, 1000]), equality_test_fun=test_nearly_equal)
    assert result != lastResult
    assert_nearly_equal(abs(result), 1.0)
    assert isinstance(result, complex)
    lastResult = result
del lastResult
assert get_normalized(0+0j) is SpecialAnswer.DNE




def multi_traverse(data, count=None):
    # itertools.product could often be used instead.
    
    # assert iter(data) is not iter(data)
    assert count >= 2
    return itertools.product(*itertools.tee(data, count))
    """
    if count == 1:
        for item in data:
            yield (item,)
    else:
        for item in data:
            for extension in multi_traverse(data, count=count-1):
                yield (item,) + extension
    """
                
assert list(multi_traverse([1,2], count=2)) == [(1,1),(1,2),(2,1),(2,2)]





def div_complex_by_i(val):
    return complex(val.imag, -val.real)

for testPt in (complex(*argPair) for argPair in multi_traverse([-100,-2,-1,0,1,2,100], count=2)):
    assert_nearly_equal(div_complex_by_i(testPt), testPt/complex(0,1))




def assure_conforms_to_clamp(value, limitPair):
    if not limitPair[0] <= value <= limitPair[1]:
        raise TestingBasics.AssuranceError("{} does not conform to clamp {}.".format(repr(value), repr(limitPair)))
    return value

for testArgs in [(0,(2.1,2.2)), (2.099999,(2.1,2.2)), (2.200001,(2.1,2.2))]:
    assert raises_instanceof(assure_conforms_to_clamp, AssuranceError, debug=True)(*testArgs), testArgs
for testArgs in [(2.1,(2.1,2.2)), (2.2,(2.1,2.2)), (2.199999,(2.1,2.2))]:
    assert not raises_instanceof(assure_conforms_to_clamp, AssuranceError, debug=False)(*testArgs), testArgs
    

def assure_exclusively_conforms_to_clamp(value, limitPair):
    if not limitPair[0] < value < limitPair[1]:
        raise TestingBasics.AssuranceError("{} does not exclusively conform to clamp {}.".format(repr(value), repr(limitPair)))
    return value

for testArgs in [(0,(2.1,2.2)), (2.099999,(2.1,2.2)), (2.1, (2.1, 2.2)), (2.200001,(2.1,2.2)), (2.2,(2.1,2.2))]:
    assert raises_instanceof(assure_exclusively_conforms_to_clamp, AssuranceError, debug=True)(*testArgs), testArgs
for testArgs in [(2.1001,(2.1,2.2)), (2.1999,(2.1,2.2))]:
    assert not raises_instanceof(assure_exclusively_conforms_to_clamp, AssuranceError, debug=False)(*testArgs), testArgs






def get_exclusive_top_right_quadrant_complex_angle(value):
    assert value.imag > 0 and value.real > 0, "{} is not in the top right quadrant.".format(value)

    result = Trig.atan(value.imag/value.real)
    assert 0.0 <= result <= Trig.half_pi
    # raise NotImplementedError("fix: sometimes answers angle of 0 or pi/2 when this should not be allowed")
    return result

# test later.
    

def get_exclusive_top_half_complex_angle(value):
    assert value.imag > 0, "{} is not in the top half.".format(value)
    if value.real == 0:
        return Trig.half_pi # there is more than one way to return pi/2 in this method, even though inputs with imag==0 are not allowed. Large number rounding errors are okay.
    else:
        if value.real < 0:
            prevQuadPoint = div_complex_by_i(value)
            assert prevQuadPoint.real > 0
            assert prevQuadPoint.imag > 0
            result = get_exclusive_top_right_quadrant_complex_angle(prevQuadPoint)
            assert 0.0 <= result <= Trig.half_pi
            return result + Trig.half_pi
        else:
            assert value.real > 0
            result = get_exclusive_top_right_quadrant_complex_angle(value)
            assert 0.0 <= result <= Trig.half_pi
            return result
    
for theta in float_range(0.1, Trig.pi-0.25, 0.01):
    for s in [0.1, 1.0, 100.0]:
        testPt = complex(Trig.cos(theta)*s, Trig.sin(theta)*s)
        if theta < Trig.half_pi:
            resultA = get_exclusive_top_right_quadrant_complex_angle(testPt)
            assert_nearly_equal(resultA, theta)
        resultB = get_exclusive_top_half_complex_angle(testPt)
        assert_nearly_equal(resultB, theta)


def get_complex_angle(point):

    if point.imag == 0:
        if point.real == 0:
            return SpecialAnswer.DNE
        elif point.real < 0:
            return Trig.pi
        else:
            assert point.real > 0
            return SpecialAnswer.ZERO_MOD_TAU
    else:

        if point.imag < 0:
            oppositePoint = point * -1
            assert oppositePoint.imag > 0
            oppositePointAngle = get_exclusive_top_half_complex_angle(oppositePoint)
            assert_equal(type(oppositePointAngle), float)
            assert 0.0 <= oppositePointAngle <= Trig.pi
            result = Trig.pi + oppositePointAngle
        else:
            assert point.imag > 0
            pointAngle = get_exclusive_top_half_complex_angle(point)
            assert 0.0 <= pointAngle <= Trig.pi
            result = pointAngle
        if result == 0.0 or result == Trig.tau: # if rounding error:
            return SpecialAnswer.ZERO_MOD_TAU
        else:
            assert 0.0 < result < Trig.tau
            return result
    assert False

lastThetaResult = None
for theta in float_range(0.1, Trig.tau+0.2, 0.01):
    thetaResult = get_shared_value((get_complex_angle(complex(Trig.cos(theta)*s, Trig.sin(theta)*s)) for s in [0.1, 1, 10, 1000]), equality_test_fun=test_nearly_equal)
    assert thetaResult != lastThetaResult
    convertedTheta = (theta % Trig.tau)
    # print((thetaResult, theta, convertedTheta, thetaResult/Trig.half_pi, theta/Trig.half_pi, convertedTheta/Trig.half_pi))
    assert_nearly_equal(thetaResult/Trig.half_pi, convertedTheta/Trig.half_pi)
    assert_nearly_equal(thetaResult, convertedTheta)
    assert isinstance(thetaResult, float)
    lastResult = thetaResult
del lastThetaResult
assert get_complex_angle(0+0j) is SpecialAnswer.DNE
assert get_complex_angle(1+0j) is SpecialAnswer.ZERO_MOD_TAU



"""
def point_polar_to_rect(polar_pt):
    return polar_pt.real*(math.e**(polar_pt.imag*1j))
    
    
def point_rect_to_polar(rect_pt):
    # assert isinstance(rect_pt, complex)
    theta = get_complex_angle(rect_pt)
    # if theta is UndefinedAtBoundary:
    #     return UndefinedAtBoundary
    return complex(abs(rect_pt), theta)
"""


def point_rect_to_polar(value):
    assert type(value) == complex, "not ready."
    if abs(value) == 0:
        # assert False, (value, abs(value))
        return SpecialAnswer.ORIGIN
    magnitude = abs(value)
    angle = get_complex_angle(value)
    if angle is SpecialAnswer.ZERO_MOD_TAU:
        return ComplexOnPolarSeam(magnitude, 0.0)
    result = complex(magnitude, angle)
    return result
    
# do tests later in load.


def point_polar_to_rect(value):
    if value is SpecialAnswer.ORIGIN:
        return complex(0,0)
    """
    assert abs(value) != 0
    assert 0 <= value.imag <= Trig.tau, value
    assert 0 < value.real, value
    """
    if isinstance(value, ComplexOnPolarSeam):
        assert value.imag == 0.0
        return complex(value.real, 0.0)
    return complex(Trig.cos(value.imag), Trig.sin(value.imag))*value.real

for real in float_range(-2.0, 2.0, 0.2):
    for imag in float_range(-2.0, 2.0, 0.2):
        assert -2 <= real <= 2
        assert -2 <= imag <= 2
        testPt = complex(real, imag)
        assert -3 < testPt.real < 3
        assert -3 < testPt.imag < 3
        if not (testPt.imag == 0 and testPt.real >= 0):
            assert point_rect_to_polar(testPt) != testPt, testPt
        assert_nearly_equal(point_polar_to_rect(point_rect_to_polar(testPt)), testPt)
        
        if (0 <= testPt.imag <= Trig.tau) and (0 <= testPt.real):
            # print(testPt)
            if testPt.imag != 0:
                _tmpConvertedPt = point_polar_to_rect(testPt)
                assert _tmpConvertedPt != testPt, (_tmpConvertedPt, testPt, real, imag)
                del _tmpConvertedPt
            polarToRectPt = point_polar_to_rect(testPt)
            assert_isinstance(polarToRectPt, complex)
            # assert abs(polarToRectPt) != 0, (polarToRectPt, type(polarToRectPt), testPt, real, imag)
            
            polarToRectToPolarPt = point_rect_to_polar(polarToRectPt)
            if testPt.real == 0:
                assert polarToRectToPolarPt == SpecialAnswer.ORIGIN, (testPt, polarToRectPt, polarToRectToPolarPt)
            elif testPt.imag == 0 and testPt.real >= 0:
                assert_equal(polarToRectToPolarPt, ComplexOnPolarSeam(testPt.real, 0.0))
            else:
                if polarToRectToPolarPt == SpecialAnswer.ORIGIN:
                    assert testPt.real == 0
                    assert polarToRectPt == 0
                assert polarToRectToPolarPt != SpecialAnswer.ORIGIN
                assert_nearly_equal(polarToRectToPolarPt, testPt)
                assert_isinstance(polarToRectToPolarPt, complex)
            
            
            
            
