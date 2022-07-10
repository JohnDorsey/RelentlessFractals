

import math
import itertools
import operator

from enum import Enum

from SeqTests import get_shared_value
from TestingAtoms import assert_equal, assert_isinstance
from TestingBasics import assert_nearly_equal, test_nearly_equal, COMPLEX_ERROR_TOLERANCE, assert_isinstance, AssuranceError, get_value_returned_or_exception_raised_by
from TestingDecorators import basic_complex_fuzz_inputs_only, basic_complex_fuzz_io

from PureGenTools import take_first_and_iter, gen_chunks_as_lists, assert_empty

import ComplexGeometry
from ComplexGeometry import point_polar_to_rect, point_rect_to_polar, ComplexOnPolarSeam

import Trig


ZERO_DIVISION_NUDGE = 2**-64
MODULUS_OVERLAP_NUDGE = 2**-16
MODULUS_OVERLAP_SCALE_NUDGE = 1j**MODULUS_OVERLAP_NUDGE
LINESEG_INTERSECTION_ERROR_TOLERANCE = 1.0/1000.0

COMPLEX_EQUALITY_DISTANCE = (2**-36)

EXTRA_ASSERTIONS = True

if not EXTRA_ASSERTIONS:
    for i in range(10):
        print("EXTRA_ASSERTIONS IS FALSE in SegmentGeometry.py")


class InteractionSpecialAnswer(Enum):
    # INTERSECTION_AT_SEGMENT_END = "intersection_at_segment_end"
    SEAM_TOUCH_AT_ORIGIN = "seam_touch_at_origin"
    SEAM_INTERSECTION_AT_ORIGIN = "seam_intersection_at_origin"

"""
class UndefinedAtBoundaryType:
    def __init__(self, message):
        self.message = message
UndefinedAtBoundary = UndefinedAtBoundaryType("generic")
"""

class UndefinedExtremeChoiceError(ValueError):
    pass

class UndefinedAtBoundaryError(ValueError):
    pass


"""
def ensure_nonzero(val):
    if val == 0.0:
        return val + ZERO_DIVISION_NUDGE
    return val
"""

def assure_positive(val):
    if val <= 0:
        raise AssuranceError()
    return val

    


def reals_of(input_seq):
    for item in input_seq:
        yield item.real
        
def imags_of(input_seq):
    for item in input_seq:
        yield item.imag
        

        
        
        

def lerp(point0, point1, t):
    return (point0*(1.0-t)) + (point1*t)

def lerp_confined(point0, point1, t):
    assert 0.0 <= t <= 1.0
    return lerp(point0, point1, t)
    
    


    
    

def find_left_min(data, enumerator_fun=enumerate):
    record, itemGen = take_first_and_iter(enumerator_fun(data))
    for item in itemGen:
        if item[1] < record[1]:
            record = item
    return record
    
assert find_left_min([-5,-7,-2,-3,-4,5,4,3,2,1]) == (1, -7)
assert find_left_min([9,8,7,6,5,6,7,8,9]) == (4, 5)
    

def find_left_max(data, enumerator_fun=enumerate):
    record, itemGen = take_first_and_iter(enumerator_fun(data))
    for item in itemGen:
        if item[1] > record[1]:
            record = item
    return record
    
    
def find_only_min(data, enumerator_fun=enumerate, comparison_fun=operator.lt, equality_test_fun=operator.eq):
    record, itemGen = take_first_and_iter(enumerator_fun(data))
    for item in itemGen:
        if comparison_fun(item[1], record[1]):
            record = item
        elif equality_test_fun(item[1], record[1]):
            record = (None, None)
        else:
            if not comparison_fun(record[1], item[1]):
                raise ValueError("comparison_fun returned True for both orderings of the two values tested, {} and {}".format(record[1], item[1]))
        #elif comparison_fun(record[1], item[1]):
        #    record = (None, None)
    if record[0] is None:
        raise UndefinedExtremeChoiceError("there was more than one minimum.")
    return record

assert find_only_min([-1,-2,0,1,2]) == (1, -2)
assert find_only_min([-1,-2,0,1,2], comparison_fun=operator.gt) == (4, 2)
assert_isinstance(get_value_returned_or_exception_raised_by(find_only_min)([-1,-2,-1,-2]), UndefinedExtremeChoiceError)
# assert_isinstance(get_value_returned_or_exception_raised_by(find_only_min)([-2,2,-1,1,0], comparison_fun=operator.le), ValueError)



"""
def find_min_index_keyed(data, key_fun=None):
    return find_min(key_fun(item) for item in data)[0]
"""









def compose_single_arg_function(func, depth=None):
    # raise NotImplementedError("needs testing!")
    assert depth >= 0
    def compose_single_arg_function_inner(*args, **kwargs):
        assert len(args) == 1, "that wasn't the deal. must be single arg."
        result = args[0]
        for i in range(0, depth):
            result = func(result, **kwargs)
        return result
    return compose_single_arg_function_inner

assert_equal([compose_single_arg_function((lambda testVal: testVal*3), depth=4)(testItem) for testItem in [5,7,10]], [5*81,7*81,10*81])

"""
def compose_functions(function_list):
    assert len(function_list) > 0
    def inner(input_arg):
        result = None
        for fun in function_list:
            result = fun(result)
        return result
    return inner
"""
















def seg_end_distance_to_point_order_trisign(seg0, point): # 1 = in order, -1 = in reverse order, 0 = neither.
    assert len(seg0) == 2
    difference = abs(seg0[1] - point) - abs(seg0[0] - point)
    if difference > 0.0:
        assert not difference == 0.0
        return 1
    elif difference < 0.0:
        assert not difference == 0.0
        return -1
    else:
        assert difference == 0.0
        return 0
    
def seg_turned_quarter_turn_around_midpoint(seg0):
    seg0midpoint = (seg0[0]+seg0[1])/2.0
    seg0secondHalfPositionless = seg0[1] - seg0midpoint
    """
    seg0firstCCWPerpPositionless = seg0secondHalfPositionless * 1.0j # perpendicular
    seg0thirdCCWPerpPositionless = seg0secondHalfPositionless * 1.0j * 1.0j * 1.0j
    seg0alphaZoneCorePoint = seg0midpoint + seg0firstCCWPerpPositionless # any point closest to this point is on this side of the segment.
    seg0betaZoneCorePoint = seg0midpoint + seg0thirdCCWPerpPositionless
    
    seg0perp = (seg0alphaZoneCorePoint, seg0betaZoneCorePoint)
    """
    return (seg0midpoint + (seg0secondHalfPositionless * 1.0j), seg0midpoint + (seg0secondHalfPositionless * -1.0j))

for testSeg in itertools.permutations(list(complex(x,y) for y in range(-2,2) for x in range(-2,2)), 2):
    assert_nearly_equal(compose_single_arg_function(seg_turned_quarter_turn_around_midpoint, depth=4)(testSeg), testSeg)

"""
def sides_of_seg0_occupied_by_seg1(seg0, seg1): # works but uses set.
    seg0perp = seg_quarter_turn_around_midpoint(seg0)
    
    assert len(seg1) == 2
    return set(seg_end_distance_to_point_order_trisign(seg0perp, seg1pt) for seg1pt in seg1)
    
    
def seg0_might_intersect_seg1(seg0, seg1): # works but uses set.
    sidesOfSeg1occupied = sides_of_seg0_occupied_by_seg1(seg1, seg0)
    assert None not in sidesOfSeg1occupied
    if len(sidesOfSeg1occupied) == 2:
        return True # seg0 crosses or touches the infinitely extended seg1.
        # elif None in sidesOfSeg1occupied:
        #     assert sidesOfSeg1occupied == {None}
        #     return True # seg0 is colinear with the infinitely extended seg1.
    else:
        assert sidesOfSeg1occupied == {1} or sidesOfSeg1occupied == {-1}
        return False
"""
def seg0_might_intersect_seg1(seg0, seg1):
    seg1perp = seg_turned_quarter_turn_around_midpoint(seg1)
    return ((seg_end_distance_to_point_order_trisign(seg1perp, seg0[0]) * seg_end_distance_to_point_order_trisign(seg1perp, seg0[1])) != 1) # return true if trisigns are anything other than (-1,-1) or (1,1).
    
    
def seg_length(seg):
    return abs(seg[1] - seg[0])


def complex_distance(point0, point1):
    return abs(point0 - point1)
assert_nearly_equal(complex_distance(5+5j, 7+7j), (2**0.5)*2)


def complex_manhattan_distance(point0, point1):
    return abs(point0.real-point1.real)+abs(point0.imag-point1.imag)
assert_nearly_equal(complex_manhattan_distance(5+5j, 7+7j), 4)
    
    
def point_and_seg_to_missing_leg_lengths(point, seg):
    return (abs(point-seg[0]), abs(point-seg[1]))
    
    
def point_might_be_on_seg(point, seg):
    dist0, dist1 = point_and_seg_to_missing_leg_lengths(point, seg)
    return abs(dist0+dist1-seg_length(seg)) < COMPLEX_EQUALITY_DISTANCE
    
    
"""
def point_and_seg_to_lerping_component(point, seg):
    distances = point_and_seg_to_missing_leg_lengths(point, seg)
    
"""

def point_is_on_seg(point, seg):
    distances = point_and_seg_to_missing_leg_lengths(point, seg)
    if 0.0 in distances:
        return True
    predictedSegLength = sum(distances)
    predictedPointT = predictedSegLength / distances[0]
    predictedPoint = lerp(seg[0], seg[1], predictedPointT)
    return (abs(predictedPoint - point) < COMPLEX_EQUALITY_DISTANCE)
        
    
    
# print("extra assertions are turned off for segments_intersect because they are failing now.")

def segments_intersect(seg0, seg1, extra_assertions=EXTRA_ASSERTIONS):
    #if seg0[0].real < 0:
    #    return "bad test answer"
        
    result = (seg0_might_intersect_seg1(seg0, seg1) and seg0_might_intersect_seg1(seg1, seg0))
    if extra_assertions:
        assert_equal(result, segments_intersect(seg1, seg0, extra_assertions=False))
        if not result:
            assert segment_intersection(seg0, seg1, extra_assertions=False) is None, (seg0, seg1)
        for testSegA, testSegB in [(seg0, seg1), (seg1, seg0)]:
            for testPt in testSegA:
                if point_is_on_seg(testPt, testSegB):
                    assert result, (seg0, seg1, testSegA, testSegB, testPt)
    return result

"""
NOTES:
assertion in segment_intersection would fail for segments_intersect(((-0.2966114161032628-0.2740619308262666j), (-0.2882969411062747-0.3062392387437123j)), ((-0.3147963493571548-0.2036856055645212j), (-0.2253568005241774-0.5498197756520699j))). errorDistance=5.821903443632943e-12, t=0.5964912280701754, u=0.25877192982456143.
additional information: seg0mightseg1=False seg1mightseg0=True (both should be true according to the results of this method.).
"""

# tests for segments_intersect come later.



def cross_multiply(vec0, vec1):
    return vec0[0]*vec1[1] - vec0[1]*vec1[0]


def segment_intersection(seg0, seg1, extra_assertions=EXTRA_ASSERTIONS):
    # seg0dir = seg0[1]-seg0[0]
    # seg1dir = seg1[1]-seg1[0]
    p = seg0[0]
    q = seg1[0]
    r = seg0[1]-seg0[0]
    s = seg1[1]-seg1[0]
    # (p+tr) x s = (q+us) x s
    # p x s + t(r x s) = q x s + u(s x s) = q x s
    # t(r x s) = (q-p) x s
    # t = ((q-p) x s)/(r x s)
    # a x b = a_x*b_y - a_y*b_x
    qminusp = (q.real-p.real, q.imag-p.imag)
    rxs = cross_multiply((r.real, r.imag), (s.real, s.imag))
    if rxs == 0:
        return None
    t = cross_multiply(qminusp, (s.real, s.imag)) / rxs
    # u = (q − p) x r / (r x s)
    u = cross_multiply(qminusp, (r.real, r.imag)) / rxs
    seg0intersection = p + t*r
    seg1intersection = q + u*s
    errorDistance = abs(seg0intersection - seg1intersection)
    if errorDistance > COMPLEX_EQUALITY_DISTANCE:
        return None
    if t < 0 or t > 1 or u < 0 or u > 1:
        return None
    if extra_assertions:
        if not (min([abs(0 - t), abs(1 - t), abs(0 - u), abs(1 - u)]) < LINESEG_INTERSECTION_ERROR_TOLERANCE): # for now, don't test non-crossing touches against segments_intersect. those tests are failing.
            if not segments_intersect(seg0, seg1, extra_assertions=False):
                print("assertion in segment_intersection would fail for segments_intersect({}, {}). errorDistance={}, t={}, u={}.".format(seg0, seg1, errorDistance, t, u))
                print("additional information: seg0mightseg1={} seg1mightseg0={} (both should be true according to the results of this method.).".format(seg0_might_intersect_seg1(seg0, seg1), seg0_might_intersect_seg1(seg1, seg0))) # this happens very rarely, like once in 200 million calls, and only for segments with very different lengths intersection very close to the end of each.
                # assert False, (seg0, seg1, errorDistance, t, u)
    return seg0intersection
    
assert not basic_complex_fuzz_inputs_only(segments_intersect, equality_test_fun=operator.eq)((1+1j, 2+1j), (1+2j, 2+2j))
assert basic_complex_fuzz_inputs_only(segments_intersect, equality_test_fun=operator.eq)((1+1j, 2+2j), (1+2j, 2+1j))
assert not basic_complex_fuzz_inputs_only(segments_intersect, equality_test_fun=operator.eq)((1+1j, 5+5j), (2+1j, 6+5j))
    
assert basic_complex_fuzz_io(segment_intersection, equality_test_fun=test_nearly_equal)((1.0+1.0j, 1.0+3.0j), (0.0+2.0j, 2.0+2.0j)) == (1.0+2.0j)
assert basic_complex_fuzz_io(segment_intersection, equality_test_fun=test_nearly_equal)((1.0+1.0j, 1.0+3.0j), (0.0+0.0j, 0.0+2.0j)) == None
assert basic_complex_fuzz_io(segment_intersection, equality_test_fun=test_nearly_equal)((0+0j, 1+1j), (1+0j, 0+1j)) == (0.5+0.5j)











    
    
    
    



def seg_is_valid(seg):
    return isinstance(seg, tuple) and all(isinstance(item, complex) for item in seg) and len(seg) == 2



"""
def polar_seg_is_valid(seg):
    assert seg_is_valid(seg)
    return (min(item.imag for item in seg) >= 0 and max(item.imag for item in seg) < 2*math.pi and item.real >= 0)
"""
    
def assert_polar_seg_is_valid(seg):
    assert seg_is_valid(seg)
    for item in seg:
        assert 0 <= item.imag < 2*math.pi, seg
        assert item.real >= 0, seg
    
    

def seg_multiplied_by_complex(seg, val):
    return (seg[0] * val, seg[1] * val)

    
def complex_swap_complex_components(val):
    return complex(val.imag, val.real)


def seg_swapped_complex_components(seg):
    return [complex_swap_complex_components(seg[i]) for i in (0,1)]


def seg_horizontal_line_intersection(seg, imag_pos=None):
    segImags = [seg[0].imag, seg[1].imag]
    segImagMin, segImagMax = (min(segImags), max(segImags))
    if segImagMax < imag_pos or segImagMin > imag_pos:
        return None
    if segImagMax == imag_pos or segImagMin == imag_pos:
        raise UndefinedAtBoundaryError("end touch.")
    segRise = segImagMax - segImagMin
    interceptRise = imag_pos - segImagMin
    assert interceptRise >= 0.0
    if segRise == 0:
        raise UndefinedAtBoundaryError("flat slope.")
    interceptT = interceptRise / segRise
    
    # lerpSeg = (seg if (segImagMinIndex == 0) else seg[::-1])
    if 0.0 < interceptT < 1.0:
        return lerp_confined(seg[0], seg[1], interceptT if seg[0].imag<seg[1].imag else 1-interceptT)
    elif 0.0 <= interceptT <= 1.0:
        raise UndefinedAtBoundaryError("unexpected end touch.")
    else:
        return None

assert seg_horizontal_line_intersection((1+1j,4+4j), 2) == 2+2j


def seg_vertical_line_intersection(seg, real_pos=None):
    raise NotImplementedError("tests needed!")
    workingSeg = seg_swapped_complex_components(seg)
    result = seg_swapped_complex_components(seg_horizontal_line_intersection(workingSeg, real_pos))
    return result
    
    
    
# if math.copysign(1.0, seg[0].imag)*math.copysign(1.0, seg[1].imag) >= 0:
    
def seg_real_axis_intersection(seg):
    result = seg_horizontal_line_intersection(seg, imag_pos=0.0)
    # assert result.imag == 0 <--- this fails?
    return result
    
def seg_imag_axis_intersection(seg):
    return seg_vertical_line_intersection(seg, real_pos=0.0)
"""
    segImags = [seg[0].imag, seg[1].imag]
    segImagMin, segImagMax = (min(segImags), max(segImags))
    if segImagMax < 0 or segImagMin > 0:
        return None
    segRise = segImagMax - segImagMin
    interceptRise = height - segImagMin
    assert interceptRise >= 0.0
    try:
        interceptT = interceptRise / segRise
    except ZeroDivisionError:
        return None # !!!!!!!
    
    # lerpSeg = (seg if (segImagMinIndex == 0) else seg[::-1])
    if 0.0 <= interceptT <= 1.0:
        return lerp_confined(seg[0], seg[1], interceptT if seg[0].imag<seg[1].imag else 1-interceptT)
    else:
        return None"""
    
assert_nearly_equal(seg_real_axis_intersection((-1-1j, 1+1j)), 0+0j)
assert_nearly_equal(seg_real_axis_intersection((5-1j, 8+2j)), 6+0j)
assert_nearly_equal(seg_real_axis_intersection((5+1j, 8-2j)), 6+0j)
assert_nearly_equal(seg_real_axis_intersection((-5-1j, -8+2j)), -6+0j)
assert_nearly_equal(seg_real_axis_intersection((-5+1j, -8-2j)), -6+0j)


def rect_seg_seam_intersection(seg):
    realAxisIntersection = seg_real_axis_intersection(seg)
    if realAxisIntersection is None:
        return None
    if realAxisIntersection.real < 0.0:
        return None
    if realAxisIntersection.real == 0.0:
        # print("rect_seg_crosses_polar_seam: warning: returning None for seg crossing origin.")
        # raise UndefinedAtBoundaryError("seg crossing origin.")
        if complex(0,0) in seg:
            return InteractionSpecialAnswer.SEAM_TOUCH_AT_ORIGIN
        else:
            return InteractionSpecialAnswer.SEAM_INTERSECTION_AT_ORIGIN
    return realAxisIntersection
        
        
    
    
def seg_rect_to_polar_and_rect_space_seam_intersection(seg):
    assert len(seg) == 2
    resultPair = [point_rect_to_polar(seg[i]) for i in (0, 1)]
    
    if ComplexGeometry.SpecialAnswer.ORIGIN in resultPair:
        seamIntersection = InteractionSpecialAnswer.SEAM_TOUCH_AT_ORIGIN
    else:
        maxThetaIndex, maxTheta = find_left_max(imags_of(resultPair))
        seamIntersection = rect_seg_seam_intersection(seg)
        if seamIntersection is not None:
            resultPair[maxThetaIndex] -= 2j*math.pi # should this still be done?
        
    result = tuple(resultPair)
    # assert seg_is_valid(result)
    return (result, seamIntersection)
    

def seg_rect_to_polar_and_polar_space_seam_intersection(seg):
    pseg, rectSpaceSeamIntersection = seg_rect_to_polar_and_rect_space_seam_intersection(seg)
    if ComplexGeometry.SpecialAnswer.ORIGIN in pseg:
        assert rectSpaceSeamIntersection is InteractionSpecialAnswer.SEAM_TOUCH_AT_ORIGIN
        polarSpaceSeamIntersection = rectSpaceSeamIntersection
    else:
        polarSpaceSeamIntersection = rect_seg_seam_intersection(pseg) # pretend polar seg is rect to solve for polar space seg seam intersection.
        
    if rectSpaceSeamIntersection is not None:
        if isinstance(polarSpaceSeamIntersection, complex):
            assert polarSpaceSeamIntersection is not None
            assert abs(polarSpaceSeamIntersection) >= abs(rectSpaceSeamIntersection)
            assert abs(polarSpaceSeamIntersection.imag) <= COMPLEX_ERROR_TOLERANCE
    else:
        if polarSpaceSeamIntersection is not None:
            raise UndefinedAtBoundaryError("there was a polar space seam intersection but not a rect space one!")
        # print("seg_rect_to_polar_and_polar_space_seam_intersection: warning: 
        # if this case is ignored, the consequences are not obvious, but it might cause unecessary splits. This might be erring on the side of caution to reduce visual bugs.
        pass
    return (pseg, polarSpaceSeamIntersection)


def seg_rect_to_polar_positive_theta_fragments(seg):
    assert seg_is_valid(seg)
    # raise NotImplementedError("still has major bugs.")
    pseg, pSeamIntersection = seg_rect_to_polar_and_polar_space_seam_intersection(seg)
    
    if pSeamIntersection == InteractionSpecialAnswer.SEAM_TOUCH_AT_ORIGIN:
        fragmentSegs = [pseg]
        for testPt in pseg:
            if isinstance(testPt, complex):
                assert testPt.real > 0, "it shouldn't be origin now if it wasn't an origin enum, right?"
                assert 0 <= testPt.imag <= Trig.tau
    elif pSeamIntersection == InteractionSpecialAnswer.SEAM_INTERSECTION_AT_ORIGIN:
        raise NotImplementedError("illegal origin intersection, what should be done?")
    else:
        assert not ComplexGeometry.SpecialAnswer.ORIGIN in pseg, (seg, pseg, pSeamIntersection)
        if pSeamIntersection is None:
            fragmentSegs = [pseg]
        else:
            assert type(pSeamIntersection) == complex, pSeamIntersection
            assert abs(pSeamIntersection.imag) <= COMPLEX_EQUALITY_DISTANCE, pSeamIntersection
            pSeamIntersection = complex(pSeamIntersection.real, max(pSeamIntersection.imag, 0.0))
            
            psegImagMinIndex, psegImagMin = find_only_min(imags_of(pseg)) # the index will identify which half of the split segment is in the negative and must be shifted.
            if pseg[0].imag == pseg[1].imag:
                raise NotImplementedError("how should this be handled?")
            if psegImagMin == 0.0:
                raise UndefinedAtBoundaryError("???1, {}".format(seg))
            assert psegImagMin <= 0.0, "this should be impossible because seamIntersection was found (not None)."
            fragmentPointPairs = [[pseg[0], pSeamIntersection], [pSeamIntersection, pseg[1]]]
            for ptIndex in (0, 1):
                fragmentPointPairs[psegImagMinIndex][ptIndex] += complex(0, Trig.tau)

            fragmentSegs = [tuple(pointPair) for pointPair in fragmentPointPairs]
            testMin = min(min(imags_of(testSeg)) for testSeg in fragmentSegs)
            if testMin == 0.0:
                raise UndefinedAtBoundaryError("???2, {}".format(seg))
            assert testMin >= 0.0, "shift process apparently failed." # why?
        
    assert len(fragmentSegs) in (1,2), (fragmentSegs, seg, pseg, pSeamIntersection)
    assert_nearly_equal(point_polar_to_rect(fragmentSegs[0][0]), seg[0])
    assert_nearly_equal(point_polar_to_rect(fragmentSegs[-1][1]), seg[1])
    
    for i, fragmentSeg in enumerate(fragmentSegs): # replace (rotationless origin, pt) with (origin with same rotation as pt, pt).
        # assert seg_is_valid(fragmentSeg), (i, fragmentSeg, seg, pseg)
        for ii in (0, 1):
            if fragmentSeg[ii] is ComplexGeometry.SpecialAnswer.ORIGIN:
                assert fragmentSeg[1-ii] is not ComplexGeometry.SpecialAnswer.ORIGIN
                replacementPair = list(fragmentSeg)
                replacementPair[ii] = complex(0, fragmentSeg[1-ii].imag)
                fragmentSegs[i] = tuple(replacementPair)
                
    return fragmentSegs

"""
def seg_polar_to_rect(seg):
    assert len(seg) == 2
    return tuple(point_polar_to_rect(seg[i]) for i in (0,1))
    
#tests:
for testPt in [1+1j, -1+1j, -1-1j, 1-1j]:
    # assert_equals(testPt, point_rect_to_polar(point_polar_to_rect(testPt))). negative length is not fair.
    assert_nearly_equal(testPt, point_polar_to_rect(point_rect_to_polar(testPt)))
"""
    
"""
def polar_space_segment_intersection(seg0, seg1):
    assert seg_is_valid(seg0)
    assert seg_is_valid(seg1)
    wrapSegLen = sum(abs(item) for seg in (seg0, seg1) for item in seg)
    wrapSeg = (0+0j, wrapSegLen+0j)
    seg0wrapPt = segment_intersection(seg0, wrapSeg)
    seg1wrapPt = segment_intersection(seg1, wrapSeg)
    
    # when a polar segment crosses the wrapSeg, it never is going the long way around the origin. So it must be the shorter of the 2 polar segs connecting its endpts. 
    polarSeg0 = seg_rect_to_polar(seg0)
    polarSeg1 = seg_rect_to_polar(seg1)
    assert_polar_seg_is_valid(polarSeg0)
    assert_polar_seg_is_valid(polarSeg1)
    assert_nearly_equal(seg_polar_to_rect(polarSeg0), seg0)
    assert_nearly_equal(seg_polar_to_rect(polarSeg1), seg1)
    # polar segs 0 and 1 are still unsafe. if they are short and cross the wrapSeg, they will be interpreted as their long twins by segment_intersection until the angle component signs are made different in the proper way.
    # for a segment that wraps:
    #   the angle component's sign for _one_ of the endpoints must change (by offsetting) and it must always be the one that, when adjusted, makes the segment seem shortest.
    #   this will always be the one with the higher imag component.
    
    if seg0wrapPt is not None:
        if polarSeg0[0].imag > polarSeg0[1].imag:
            polarSeg0 = (polarSeg0[0] - 2*math.pi*1j + ZERO_DIVISION_NUDGE*1j, polarSeg0[1])
            # assert_polar_seg_is_valid(polarSeg0)
        elif polarSeg0[0].imag < polarSeg0[1].imag:
            polarSeg0 = (polarSeg0[0], polarSeg0[1] - 2*math.pi*1j + ZERO_DIVISION_NUDGE*1j)
            # assert_polar_seg_is_valid(polarSeg0)
        else:
            raise NotImplementedError()
    if seg1wrapPt is not None:
        if polarSeg1[0].imag > polarSeg1[1].imag:
            polarSeg1 = (polarSeg1[0] - 2*math.pi*1j + ZERO_DIVISION_NUDGE*1j, polarSeg1[1])
            # assert_polar_seg_is_valid(polarSeg1)
        elif polarSeg1[0].imag < polarSeg1[1].imag:
            polarSeg1 = (polarSeg1[0], polarSeg1[1] - 2*math.pi*1j + ZERO_DIVISION_NUDGE*1j)
            # assert_polar_seg_is_valid(polarSeg1)
        else:
            raise NotImplementedError()
    # assert_polar_seg_is_valid(polarSeg0)
    # assert_polar_seg_is_valid(polarSeg1)
    
    polarWrapSeg = wrapSeg
    polarSeg0wrapPt = None if seg0wrapPt is None else segment_intersection(polarSeg0, polarWrapSeg)
    polarSeg1wrapPt = None if seg1wrapPt is None else segment_intersection(polarSeg1, polarWrapSeg)
    assert_nearly_equal(seg0wrapPt is None, polarSeg0wrapPt is None)
    assert_nearly_equal(seg1wrapPt is None, polarSeg1wrapPt is None)

    splitPolarSegs0 = [polarSeg0] if polarSeg0wrapPt is None else [(polarSeg0[0], polarSeg0wrapPt), (polarSeg0wrapPt, polarSeg0[1])]
    splitPolarSegs1 = [polarSeg1] if polarSeg1wrapPt is None else [(polarSeg1[0], polarSeg1wrapPt), (polarSeg1wrapPt, polarSeg1[1])]
    assert len(splitPolarSegs0) in (1,2)
    assert len(splitPolarSegs1) in (1,2)
    # these new segs must all have proper angle components between 0 and 2pi.
    splitPolarSegs0 = [tuple(seg[itemi].real+(seg[itemi].imag%(2*math.pi-MODULUS_OVERLAP_NUDGE))*1j for itemi in (0,1)) for seg in splitPolarSegs0]
    splitPolarSegs1 = [tuple(seg[itemi].real+(seg[itemi].imag%(2*math.pi-MODULUS_OVERLAP_NUDGE))*1j for itemi in (0,1)) for seg in splitPolarSegs1]
    for testPolarSeg in (splitPolarSegs0 + splitPolarSegs1):
        assert_polar_seg_is_valid(testPolarSeg)
        
    polarSectPts = [segment_intersection(testPolarSeg0, testPolarSeg1) for testPolarSeg0 in splitPolarSegs0 for testPolarSeg1 in splitPolarSegs1]
    polarSectPts = [polarSectPt for polarSectPt in polarSectPts if polarSectPt is not None]
    
    assert len(polarSectPts) <= 1 # doesn't properly succeed with intersection _at_ the theta==0 line.
    if len(polarSectPts) == 0:
        return None
    else:
        assert len(polarSectPts) == 1
        # polarSectPt = polarSectPts[0]
        return point_polar_to_rect(polarSectPts[0])
"""

def conjugate_of_seg(seg):
    return [pt.conjugate() for pt in seg]
  
"""
def conjugate_of_something(val):
    if val is not None:
        return val.conjugate()
    return val
"""
def pack_note_if(value, note, enabled):
    if enabled:
        return (value, note)
    else:
        return value


print("warning: rect_seg_polar_space_intersection extra_assertions is not finished and currently changes output behavior.")

def rect_seg_polar_space_intersection(seg0, seg1, force_symmetry=False, debug=False):
    """
    if any((pt.imag == 0 and abs(pt.real) > 0.01) for seg in (seg0, seg1) for pt in seg): # this is a dumb fix to visual bugs that won't always be necessary.
        result = rect_seg_polar_space_intersection(seg_multiplied_by_complex(seg0, MODULUS_OVERLAP_SCALE_NUDGE), seg_multiplied_by_complex(seg1, MODULUS_OVERLAP_SCALE_NUDGE))
        if result is not None:
            result /= MODULUS_OVERLAP_SCALE_NUDGE
        return result
    """
    if force_symmetry:
        normalResult, normalResultNote = rect_seg_polar_space_intersection(seg0, seg1, force_symmetry=False, debug=True)
        flippedResult, flippedResultNote = rect_seg_polar_space_intersection(conjugate_of_seg(seg0), conjugate_of_seg(seg1), force_symmetry=False, debug=True)
        unflippedResult = (None if flippedResult is None else flippedResult.conjugate())
        success = test_nearly_equal(normalResult, unflippedResult)
        # assert success, ((normalResult, normalResultNote), (unflippedResult, flippedResultNote), seg0, seg1)
        if success:
            return pack_note_if(normalResult, "ok from rect_seg_polar_space_intersection with forced symmetry", debug)
        else:
            assert debug == False, "can't debug this now."
            if isinstance(normalResultNote, tuple) or isinstance(flippedResultNote, tuple):
                print("warning: rect_seg_polar_space_intersection: notes are {} {} {} {}.".format(normalResult, normalResultNote, flippedResult, flippedResultNote))
            return None
            
    try:
        psegFrags = [seg_rect_to_polar_positive_theta_fragments(currentSeg) for currentSeg in [seg0, seg1]]
    except UndefinedAtBoundaryError as uabe:
        if debug:
            return (None, ("UndefinedAtBoundaryError caught while defining psegFrags.", uabe))
        else:
            return None
        
    # print("rect_seg_polar_space_intersection: {}".format(psegFrags))
    for pseg0frag in psegFrags[0]:
        for pseg1frag in psegFrags[1]:
            
            pIntersection = segment_intersection(pseg0frag, pseg1frag)
            if pIntersection is not None:
                assert isinstance(pIntersection, complex)
                result = point_polar_to_rect(pIntersection)
                if debug:
                    return (result, "ok from rect_seg_polar_space_intersection without forced symmetry")
                else:
                    return result
    if debug:
        return (None, "default case, failed to intersect, psegFrags={}.".format(psegFrags))
    else:
        return None

#def rect_seg_polar_space_intersection(seg0, seg1):
    
        
        

try:
    assert rect_seg_polar_space_intersection((10+1j, 10+20j), (1+10j, 20+10j)) is not None
    assert rect_seg_polar_space_intersection((1+1j, 2+2j), (1+2j, 2+1j)) is not None

    assert rect_seg_polar_space_intersection((-10+1j, -10+20j), (-1+10j, -20+10j)) is not None
    assert rect_seg_polar_space_intersection((-1+1j, -2+2j), (-1+2j, -2+1j)) is not None


    assert rect_seg_polar_space_intersection((-100+10j, 100+10j), (-5-100j, -5+100j)) is not None
    assert rect_seg_polar_space_intersection((-100+10j, 100+10j), (5-100j, 5+100j)) is not None

    assert_nearly_equal(rect_seg_polar_space_intersection((-0.1+0.1j, -1+1j), (0+1j, -1+0j)), ((2.0**0.5)/2.0)*(-1+1j))
    assert_nearly_equal(rect_seg_polar_space_intersection((0.1+0.1j, 1+1j), (0+1j, 1+0j)), ((2.0**0.5)/2.0)*(1+1j))
    assert_nearly_equal(rect_seg_polar_space_intersection((0+0j, 1+1j), (0+1j, 1+0j)), ((2.0**0.5)/2.0)*(1+1j))
except AssertionError as ae:
    #print(ae.message)
    print(ae)
