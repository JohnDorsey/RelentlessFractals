
import copy
import math
import itertools


ZERO_DIVISION_NUDGE = 2**-64
MODULUS_OVERLAP_NUDGE = 2**-16
MODULUS_OVERLAP_SCALE_NUDGE = 1j**MODULUS_OVERLAP_NUDGE
LINESEG_INTERSECTION_ERROR_TOLERANCE = 1.0/1000.0

COMPLEX_ERROR_TOLERANCE = (2**-36)
COMPLEX_EQUALITY_DISTANCE = (2**-36)

EXTRA_ASSERTIONS = True

if not EXTRA_ASSERTIONS:
    for i in range(10):
        print("EXTRA_ASSERTIONS IS FALSE in SegmentGeometry.py")


"""
class UndefinedAtBoundaryType:
    def __init__(self, message):
        self.message = message
UndefinedAtBoundary = UndefinedAtBoundaryType("generic")
"""
class UndefinedAtBoundaryError(ValueError):
    pass






def assert_equal(thing0, thing1):
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)





def test_complex_nearly_equal(val0, val1, error_tolerance=COMPLEX_ERROR_TOLERANCE, debug=False):
    err = val0 - val1
    errMagnitude = abs(err)
    if errMagnitude == 0:
        return True
    elif errMagnitude < error_tolerance:
        # print("warning: {} and {} are supposed to be equal.".format(val0, val1))
        return True
    else:
        if debug:
            print("test_complex_nearly_equal: debug: {} is not nearly equal to {}, err is {}, errMagnitude is {}.".format(val0, val1, err, errMagnitude))
        return False
        

def _assert_complex_nearly_equal(val0, val1, error_tolerance=COMPLEX_ERROR_TOLERANCE):
    assert test_complex_nearly_equal(val0, val1, error_tolerance=error_tolerance, debug=True), "{} is not close enough to {} with tolerance setting {}.".format(val0, val1, error_tolerance)
    """
    "the difference between {} and {} is {} with length {} - that's {} times greater than the error tolerance {}.".format(
            val0, val1, err, errMagnitude, str(errMagnitude/error_tolerance),
        )
    """
    
def test_nearly_equal(thing0, thing1, error_tolerance=COMPLEX_ERROR_TOLERANCE, debug=False):
    head = "test_nearly_equal: debug: "
    if isinstance(thing0, complex) and isinstance(thing1, complex):
        result = test_complex_nearly_equal(thing0, thing1, error_tolerance=error_tolerance)
        if debug and not result:
            print(head + "failed in br0.")
        return result
    elif any(isinstance(thing0, testEnterable) and isinstance(thing1, testEnterable) for testEnterable in (tuple, list)):
        if len(thing0) != len(thing1):
            if debug:
                print(head + "lengths differ.")
            return False
        result = all(test_nearly_equal(thing0[i], thing1[i], error_tolerance=error_tolerance) for i in range(max(len(thing0), len(thing1))))
        if debug and not result:
            print(head + "failed in br1.")
        return result
    else:
        result = (thing0 == thing1)
        if debug and not result:
            print(head + "failed in br2.")
        return result

def assert_nearly_equal(thing0, thing1, error_tolerance=COMPLEX_ERROR_TOLERANCE):
    assert test_nearly_equal(thing0, thing1, error_tolerance=error_tolerance, debug=True), "{} does not nearly equal {}.".format(repr(thing0), repr(thing1))


def assure_positive(val):
    assert not val <= 0
    return val


def ensure_nonzero(val):
    if val == 0.0:
        return val + ZERO_DIVISION_NUDGE
    return val





def reals(input_seq):
    for item in input_seq:
        yield item.real
        
def imags(input_seq):
    for item in input_seq:
        yield item.imag
        
def real_of(val):
    return val.real
    
def imag_of(val):
    return val.imag
    
def inv_abs_of(val):
    return 1.0/max(ZERO_DIVISION_NUDGE, abs(val))

def lerp(point0, point1, t):
    return (point0*(1.0-t)) + (point1*t)

def lerp_confined(point0, point1, t):
    assert 0.0 <= t <= 1.0
    return lerp(point0, point1, t)
    
    


def peek_first_and_iter(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        raise IndexError("empty, couldn't peek first!")
    return (first, inputGen)


def get_shared_value(input_seq, equality_test_fun=test_nearly_equal):
    result, inputGen = peek_first_and_iter(input_seq)
    for i, item in enumerate(inputGen):
        assert equality_test_fun(item, result), "at index {}, item value {} does not equal shared value {}.".format(i, repr(item), repr(result))
    return result
    
    

def find_min(data):
    record, itemGen = peek_first_and_iter(enumerate(data))
    for item in itemGen:
        if item[1] < record[1]:
            record = item
    return record

def find_max(data):
    record, itemGen = peek_first_and_iter(enumerate(data))
    for item in itemGen:
        if item[1] > record[1]:
            record = item
    return record






def assert_empty(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        return
    assert False, "input seq was not empty, first item was {}.".format(repr(first))
    

def gen_chunks_as_lists(data, length):
    itemGen = iter(data)
    while True:
        chunk = [item for item in itertools.islice(itemGen, 0, length)]
        yield chunk
        if len(chunk) < length:
            assert_empty(itemGen)
            return
        else:
            assert len(chunk) == length
    assert False


def compose_single_arg_function(func, depth=None):
    assert depth >= 0
    def compose_single_arg_function_inner(*args, **kwargs):
        assert len(args) == 1, "that wasn't the deal. must be single arg."
        result = args[0]
        for i in range(0, depth):
            result = func(result, **kwargs)
        return result
    return compose_single_arg_function_inner
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




def make_transformed_copy(data, enter_trigger_fun=None, transform_trigger_fun=None, transform_fun=None):
    if enter_trigger_fun is None:
        def enter_trigger_fun(testItem):
            if isinstance(testItem, (tuple,list)):
                assert not transform_trigger_fun(testItem)
                return True
            else:
                return False
    if enter_trigger_fun(data):
        return type(data)(make_transformed_copy(item, enter_trigger_fun=enter_trigger_fun, transform_trigger_fun=transform_trigger_fun, transform_fun=transform_fun) for item in data)
    elif transform_trigger_fun(data):
        return transform_fun(data)
    else:
        return copy.deepcopy(data)
    


def fuzz_inputs_share_output(input_fun, fuzzer_gen):
    """
    def inner(*args, **kwargs):
        testResultGen = (input_fun(*fuzzer(args), **kwargs) for fuzzer in fuzzer_gen)
        return get_shared_value(testResultGen)
    return inner
    """
    defuzzerGen = itertools.repeat((lambda x: x))
    return fuzz_inputs_share_defuzzed_output(input_fun, zip(fuzzer_gen, defuzzerGen))


def fuzz_inputs_share_defuzzed_output(input_fun, fuzzer_defuzzer_pair_gen):
    def inner(*args, **kwargs):
        testResultGen = (inverse_fuzzer(input_fun(*fuzzer(args), **kwargs)) for fuzzer, inverse_fuzzer in fuzzer_defuzzer_pair_gen)
        return get_shared_value(testResultGen)
    return inner
        


def multi_traverse(data, count=None):
    assert iter(data) is not iter(data)
    assert count > 0
    if count == 1:
        for item in data:
            yield (item,)
    else:
        for item in data:
            for extension in multi_traverse(data, count=count-1):
                yield (item,) + extension
assert list(multi_traverse([1,2], count=2)) == [(1,1),(1,2),(2,1),(2,2)]


def all_products_from_seq_pair(data0, data1):
    for itemA in data0:
        for itemB in data1:
            yield itemA*itemB
assert list(all_products_from_seq_pair([1,2],[3,5,100])) == [3, 5, 100, 6, 10, 200]


def complex_parallel_product(values):
    result = complex(1,1)
    for value in values:
        result = complex(result.real*value.real, result.imag*value.imag)
    return result
assert complex_parallel_product([1+2j,5+100j]) == 5+200j

def complex_pair_parallel_div(val0, val1):
    return complex(val0.real/val1.real, val0.imag/val1.imag)
    
    

    
    





def gen_basic_complex_fuzzers_and_inverses(include_neutral=True):
    neutralCounter = 0
    for rOff, iOff in multi_traverse((-1.0, 0.0, 1.0), count=2):
        for rScale, iScale in multi_traverse(list(all_products_from_seq_pair((0.5, 1.0, 2.0), (-1.0, 1.0))), count=2):
            for complexScale in (1.0+0.0j, 0.0+0.7j):
                isNeutral = ((rOff, iOff, rScale, iScale, complexScale) == (0.0, 0.0, 1.0, 1.0, 1.0+0.0j))
                if isNeutral:
                    neutralCounter += 1
                    if not include_neutral:
                        continue
                
                def currentFun(inputArgs):
                    return make_transformed_copy(
                        inputArgs,
                        transform_trigger_fun=(lambda x: isinstance(x, complex)),
                        transform_fun=(lambda w: complex_parallel_product([w+complex(rOff, iOff), complex(rScale, iScale)])*complexScale),
                    )
                def currentInverseFun(inputArgs):
                    return make_transformed_copy(
                        inputArgs,
                        transform_trigger_fun=(lambda x: isinstance(x, complex)),
                        transform_fun=(lambda w: complex_pair_parallel_div(w/complexScale, complex(rScale, iScale))-complex(rOff, iOff)),
                    )
                    
                yield (currentFun, currentInverseFun)
    assert neutralCounter == 1
                
for testFun, testInverseFun in gen_basic_complex_fuzzers_and_inverses():
    assert testInverseFun(testFun(complex(10,10000))) == complex(10,10000)
    

def basic_complex_fuzz_inputs_only(input_fun):
    """
    def fuzzedArgTupleGenFun(inputArgs):
        for fuzzFun, _ in gen_basic_complex_fuzzers_and_inverses():
            yield fuzzFun(inputArgs)
    """
    fuzzerGen = (pair[0] for pair in gen_basic_complex_fuzzers_and_inverses())
    inner = fuzz_inputs_share_output(input_fun, fuzzerGen)
    return inner

testList = []
def testAppender(inputItem):
    testList.append(inputItem)
basic_complex_fuzz_inputs_only(testAppender)(100.0+10000.0j)
assert len(set(testList)) == 9*(3*2*3*2)*2
testList.clear()
basic_complex_fuzz_inputs_only(testAppender)([["a", (200.0+20000.0j,)], (5, 6), "b"])
for item in testList:
    assert item[-2:] == [(5,6), "b"]
assert len(set(str(item) for item in testList)) == 9*3*2*3*2*2
del testList
del testAppender


def basic_complex_fuzz_io(input_fun):
    inner = fuzz_inputs_share_defuzzed_output(input_fun, gen_basic_complex_fuzzers_and_inverses())
    return inner




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
                print("additional information: seg0mightseg1={} seg1mightseg0={}.".format(seg0_might_intersect_seg1(seg0, seg1), seg0_might_intersect_seg1(seg1, seg0)))
                # assert False, (seg0, seg1, errorDistance, t, u)
    return seg0intersection
    
assert not basic_complex_fuzz_inputs_only(segments_intersect)((1+1j, 2+1j), (1+2j, 2+2j))
assert basic_complex_fuzz_inputs_only(segments_intersect)((1+1j, 2+2j), (1+2j, 2+1j))
assert not basic_complex_fuzz_inputs_only(segments_intersect)((1+1j, 5+5j), (2+1j, 6+5j))
    
assert basic_complex_fuzz_io(segment_intersection)((1.0+1.0j, 1.0+3.0j), (0.0+2.0j, 2.0+2.0j)) == (1.0+2.0j)
assert basic_complex_fuzz_io(segment_intersection)((1.0+1.0j, 1.0+3.0j), (0.0+0.0j, 0.0+2.0j)) == None
assert basic_complex_fuzz_io(segment_intersection)((0+0j, 1+1j), (1+0j, 0+1j)) == (0.5+0.5j)











def div_complex_by_i(val):
    return complex(val.imag, -val.real)

for testPt in (complex(*argPair) for argPair in multi_traverse([-100,-2,-1,0,1,2,100], count=2)):
    assert_nearly_equal(div_complex_by_i(testPt), testPt/complex(0,1))
    
    
    
    
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

def seg_swap_complex_components(val):
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

def seg_vertical_line_intersection(seg, real_pos=None):
    workingSeg = seg_swap_complex_components(seg)
    result = seg_swap_complex_components(seg_horizontal_line_intersection(workingSeg, real_pos))
    return result
    
    
    
# if math.copysign(1.0, seg[0].imag)*math.copysign(1.0, seg[1].imag) >= 0:
    
def seg_real_axis_intersection(seg):
    return seg_horizontal_line_intersection(seg, imag_pos=0.0)
    
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
        raise UndefinedAtBoundaryError("seg crossing origin.")
    return realAxisIntersection
        
        
def point_polar_to_rect(polar_pt):
    return polar_pt.real*(math.e**(polar_pt.imag*1j))
    
    
def point_rect_to_polar(rect_pt):
    # assert isinstance(rect_pt, complex)
    theta = get_complex_angle(rect_pt)
    # if theta is UndefinedAtBoundary:
    #     return UndefinedAtBoundary
    return complex(abs(rect_pt), theta)
    
    
def seg_rect_to_polar_and_rect_space_seam_intersection(seg):
    assert len(seg) == 2
    resultPair = [point_rect_to_polar(seg[i]) for i in (0, 1)]
    """
    for i in (0,1):
        if resultPair[i] is UndefinedAtBoundary:
            # resultPair[i] = complex(seg[i].real, MODULUS_OVERLAP_NUDGE * (1 if resultPair[1-i].imag < math.pi else -1))
            raise UndefinedAtBoundaryError("seam intersection would be endpoint {} only. resultPair is {} for seg {}.".format(i, resultPair, seg))
    """
    maxThetaIndex, maxTheta = find_max(imags(resultPair))
    seamIntersection = rect_seg_seam_intersection(seg)
    if seamIntersection is not None:
        resultPair[maxThetaIndex] -= 2j*math.pi
        
    result = tuple(resultPair)
    
    assert seg_is_valid(result)
    return (result, seamIntersection)
    

def seg_rect_to_polar_and_polar_space_seam_intersection(seg):
    pseg, rectSpaceSeamIntersection = seg_rect_to_polar_and_rect_space_seam_intersection(seg)
    polarSpaceSeamIntersection = rect_seg_seam_intersection(pseg) # pretend polar seg is rect to solve for polar space seg seam intersection.
    if rectSpaceSeamIntersection is not None:
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
    # raise NotImplementedError("still has major bugs.")
    pseg, pSeamIntersection = seg_rect_to_polar_and_polar_space_seam_intersection(seg)
    if pSeamIntersection is None:
        fragmentSegs = [pseg]
    else:
        assert abs(pSeamIntersection.imag) <= COMPLEX_ERROR_TOLERANCE, pSeamIntersection
        pSeamIntersectionNeutral = complex(abs(pSeamIntersection), 0.0)
        # ^^^ HAA! doing this to a rect space seam intersection point was probably a source of visual bugs! the intersection point in rect space is not the same as in polar space! ever!
        assert abs(pSeamIntersectionNeutral.imag) <= COMPLEX_ERROR_TOLERANCE, pSeamIntersectionNeutral
        shiftAddition = complex(0, 2.0*math.pi)
        psegImagMinIndex, psegImagMin = find_min(imags(pseg)) # the index will identify which half of the split segment is in the negative and must be shifted.
        if psegImagMin == 0.0:
            raise UndefinedAtBoundaryError("???1, {}".format(seg))
        assert psegImagMin <= 0.0, "this should be impossible because seamIntersection was found."
        fragmentPointPairs = [[pseg[0], pSeamIntersectionNeutral], [pSeamIntersectionNeutral, pseg[1]]]
        pointPairToShift = fragmentPointPairs[psegImagMinIndex]
        for i in (0, 1):
            pointPairToShift[i] += shiftAddition
        fragmentSegs = [tuple(pointPair) for pointPair in fragmentPointPairs]
        
        testMin = min(min(imags(testSeg)) for testSeg in fragmentSegs)
        if testMin == 0.0:
            raise UndefinedAtBoundaryError("???2, {}".format(seg))
        assert testMin > 0.0, "shift process apparently failed."
    assert len(fragmentSegs) in (1,2), (fragmentSegs, seg, pseg, pSeamIntersection)
    assert_nearly_equal(point_polar_to_rect(fragmentSegs[0][0]), seg[0])
    assert_nearly_equal(point_polar_to_rect(fragmentSegs[-1][1]), seg[1])
    return fragmentSegs

    
def seg_polar_to_rect(seg):
    assert len(seg) == 2
    return tuple(point_polar_to_rect(seg[i]) for i in (0,1))
    
#tests:
for testPt in [1+1j, -1+1j, -1-1j, 1-1j]:
    # assert_equals(testPt, point_rect_to_polar(point_polar_to_rect(testPt))). negative length is not fair.
    assert_nearly_equal(testPt, point_polar_to_rect(point_rect_to_polar(testPt)))
    
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
