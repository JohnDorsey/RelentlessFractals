#!/usr/bin/python3




import os; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import numpy


import time
import math
import itertools
import collections
import copy
import random
import gc

import pygame

from TestingAtoms import assert_equal, summon_cactus

try:
    import fxpmath
except ImportError:
    print("fxpmath is not installed. You might not need it, though.")
    fxpmath = summon_cactus("fxpmath_was_never_imported_because_it_is_not_installed")

from ColorTools import atan_squish_to_byteint_unsigned_uniform_nearest

from ComplexGeometry import real_of, imag_of, inv_abs_of, get_complex_angle, get_normalized, float_range
import SegmentGeometry
from SegmentGeometry import find_left_min, lerp, reals_of, imags_of

import ComplexGeometry

from PureGenTools import gen_track_previous, take_first_and_iter, gen_track_previous_full, gen_track_recent, ProvisionError, izip_longest, gen_track_recent_trimmed, enumerate_to_depth_packed, iterate_to_depth, izip_shortest, gen_chunks_as_lists
from HigherRangeFunctionalTools import higher_range, higher_range_by_corners, corners_to_range_descriptions

import Trig
sin, cos, tan = (Trig.sin, Trig.cos, Trig.tan) # short names for use only in compilation of mandel methods.
cpx, norm = (complex, get_normalized)

import CGOL
import MatrixMath




import PygameDashboard
from PygameDashboard import measure_time_nicknamed


def THIS_MODULE_EXEC(string):
    exec(string)

CAPTION_RATE_LIMITER = PygameDashboard.SimpleRateLimiter(1.0)
STATUS_RATE_LIMITER = PygameDashboard.RateLimiter(3.0)
PASSIVE_DISPLAY_FLIP_RATE_LIMITER = PygameDashboard.RateLimiter(30.0)

COMPLEX_NAN = complex(math.nan, math.nan)








def shape_of(data_to_test):
    result = []
    while hasattr(data_to_test, "__len__"):
        if isinstance(data_to_test, str):
            print("shape_of: warning: a string will not be treated as a storage object, but this behavior is not standard.")
            break
        result.append(len(data_to_test))
        if result[-1] == 0:
            break
        data_to_test = data_to_test[0]
    return tuple(result)
    
    

    
    
    
    
    
    

def enumerate_from_both_ends(data):
    assert hasattr(data, "__len__")
    for forwardI, item in enumerate(data):
        yield (forwardI, len(data)-forwardI-1, item)
            
assert [item for item in enumerate_from_both_ends("abc")] == [(0,2,"a"), (1,1,"b"), (2,0,"c")]
assert [item for item in enumerate_from_both_ends("abcd")] == [(0,3,"a"), (1,2,"b"), (2,1,"c"), (3,0,"d")]
            


def is_round_binary(value):
    assert value > 0
    return value == 2**(value.bit_length()-1)
    
assert all(is_round_binary(testNum) for testNum in [2**i for i in range(2, 33)])
assert not any(is_round_binary(testNum) for testNum in [2**i+chg for i in range(2,33) for chg in (-1,1)])


def assure_round_binary(value):
    assert is_round_binary(value), "could not assure value is round in binary."
    return value


def enforce_tuple_length(input_tuple, length, default=None):
    assert type(input_tuple) == tuple
    if len(input_tuple) == length:
        return input_tuple
    elif len(input_tuple) < length:
        return input_tuple + tuple(default for i in range(length-len(input_tuple)))
    else:
        return input_tuple[:length]


    
    
    



def to_portable(path_str):
    # https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
    # windows forbidden: "<>:\"/\\|?*"
    # slashes are allowed because they are used for saving to a folder.
    """
    forbiddenDict = {
        "<":"Lss", ">":"Gtr", ":":"Cln", "\"":"Dblqt", "\\":"Bkslsh", "|":"Vrtpipe", "?":"Quemrk", "*":"Astrsk", 
        "=":"Eq", "'":"Sglqt", "!":"Exclmrk", "@":"Atsgn", "#":"Poundsgn", "$":"Dlrsgn", "%":"Prcntsgn",  "^":"Caret", "&":"Amprsnd",
    }
    """
    forbiddenDict = {
        "<":"LS", ">":"GR", ":":"CN", "\"":"DQ", "\\":"BS", "|":"VP", "?":"QM", "*":"AK", 
        "=":"EQ", "'":"SQ", "!":"XM", "@":"AT", "#":"HS", "$":"DS", "%":"PC",  "^":"CT", "&":"AP",
        ";":"SN", "~":"TD", "[":"LB", "]":"RB", "{":"LC", "}":"RC",
    }
    for forbiddenChar, replacementChar in forbiddenDict.items():
        oldPathStr = path_str
        path_str = path_str.replace(forbiddenChar, replacementChar)
        if path_str != oldPathStr:
            print("to_portable: Warning: ~{} occurrences of {} will be replaced with {} for portability.".format(oldPathStr.count(forbiddenChar), repr(forbiddenChar), repr(replacementChar)))
    return path_str










@measure_time_nicknamed("save_surface_as", end="\n\n", include_lap=True, include_load=True)
def save_surface_as(surface, name_prefix="", name=None,
    _gccollect = measure_time_nicknamed("garbage collection", include_load=True)(gc.collect)
):
    if name is None:
        size = surface.get_size()
        sizeStr = str(size).replace(", ","x") if (size[0] != size[1]) else "({}x)".format(size[0])
        name = "{}{}.png".format(round(time.monotonic(), ndigits=1), sizeStr)
    usedName = to_portable(OUTPUT_FOLDER + name_prefix + name)
    print("saving file {}.".format(usedName))
    assert usedName.endswith(".png")
    pygame.image.save(surface, usedName)
    _gccollect()
    #print("{} unreachable objects.".format())



    



@measure_time_nicknamed("draw_squished_ints_to_surface", include_load=True)
def draw_squished_ints_to_surface(dest_surface, channels, access_order=None):
    # maybe this method shouldn't exist. Maybe image creation should happen in another process, like photo.py in GeodeFractals.
    assert COLOR_SETTINGS_SUMMARY_STR == "color(atan)"
    
    if access_order == "cyx":
        colorDataGetter = lambda argX, argY, argC: channels[argC][argY][argX]
        xSize, ySize, cSize = (len(channels[0][0]), len(channels[0]), len(channels))
    elif access_order == "yxc":
        colorDataGetter = lambda argX, argY, argC: channels[argY][argX][argC]
        xSize, ySize, cSize = (len(channels[0]), len(channels), len(channels[0][0]))
    else:
        raise ValueError("unsupported access order.")
    assert cSize == 3, shape_of(channels)
    assert xSize > 4
    assert ySize > 4
        
    try:
        for y in range(ySize):
            for x in range(xSize):
                color = tuple(atan_squish_to_byteint_unsigned_uniform_nearest(colorDataGetter(x, y, chi)) for chi in range(cSize))
                dest_surface.set_at((x, y), color)
    except IndexError as ie:
        print("index error when (x, y)=({}, {}): {}.".format(x, y, ie))
        exit(1)
            

            








def dv1range(subdivisions):
    denom = float(subdivisions)
    for x in range(subdivisions):
        yield x/denom

assert_equal(list(dv1range(2)), [0.0, 0.5])
assert_equal(list(dv1range(4)), [0.0, 0.25, 0.5, 0.75])


            
def construct_data(size, default_value=None, converter_fun=None, print_status=False):
    assert len(size) > 0
    if converter_fun is not None:
        raise NotImplementedError("converter_fun")
    """
    if STATUS_RATE_LIMITER.get_judgement():
        print()
    """
    if len(size) == 1:
        result = [copy.deepcopy(default_value) for i in range(size[0])]
        return result
    else:
        result = []
        for i in range(size[0]):
            if STATUS_RATE_LIMITER.get_judgement():
                print("construct_data: {}%...".format(round(i*100.0/size[0], ndigits=3)))
            result.append(construct_data(size[1:], default_value=default_value, print_status=False))
        return result

assert_equal(shape_of(construct_data([5,6,7])), (5,6,7))


def construct_numpy_data(size, default_value=None):
    return construct_data(size, default_value=default_value, converter_fun=numpy.array)



def fill_data(data, fill_value):
    # print("fix fill data")
    assert isinstance(data, (list, numpy.ndarray)), type(data)
    for i in range(len(data)):
        if isinstance(data[i], (list, numpy.ndarray)):
            fill_data(data[i], fill_value)
        elif isinstance(data[i], (tuple, str)):
            raise TypeError("type {} can't be processed!".format(type(data[i])))
        else:
            if not hasattr(data[i], "__setitem__") == hasattr(fill_value, "__setitem__"): # don't test for __getitem__ because numpy.int64 has that.
                raise NotImplementedError("type can't be changed from {} to {} because only one seems to be a container!".format(repr(type(data[i])), repr(type(fill_value))))
            data[i] = fill_value



def gen_assuredly_ascending(input_seq):
    for previousItem, item in gen_track_previous(input_seq):
        if previousItem is not None:
            assert previousItem <= item , "could not assure ascending! not {} <= {}.".format(repr(previousItem),repr(item))
        yield item



"""
def mutate_method_consts(fun_to_mutate, replacement_dict):
    originalConsts = fun_to_mutate.__code__.co_consts
    for key in replacement_dict.keys():
        assert originalConsts.count(key) == 1, (key, fun_to_mutate, originalConsts)
    fun_to_mutate.__code__ = fun_to_mutate.__code__.replace(co_consts=tuple((replacement_dict[item] if item in replacement_dict else item) for item in originalConsts))
cannot work. modifies original even if deepcopies are made.
"""













_mandelMethodsSourceStrs={
        "c_to_mandel_itercount_fast":"""
def c_to_mandel_itercount_fast(c, iter_limit):
    ${init_formula}
    for n in range(iter_limit):
        if ${esc_test}:
            return n
        ${iter_formula}
    return None""",
    
        "c_to_escstop_mandel_journey":"""
def c_to_escstop_mandel_journey(c):
    ${init_formula}
    for n in itertools.count():
        ${yield_formula}
        if ${esc_test}:
            return
        ${iter_formula}""",
        
    }
    
# z0="0+0j", exponent="2"
def compile_mandel_method(method_name, init_formula=None, yield_formula=None, esc_test=None, iter_formula=None):
    sourceStr = _mandelMethodsSourceStrs[method_name]
    assert sourceStr.count("def {}(".format(method_name)) == 1, "bad source code string for name {}!".format(method_name)
    
    sourceStr = sourceStr.replace("${init_formula}", init_formula).replace("${yield_formula}", yield_formula).replace("${esc_test}", esc_test).replace("${iter_formula}", iter_formula)
    
    exec(sourceStr)
    assert method_name in locals().keys(), "method name {} wasn't in locals! bad source code string?".format(method_name)
    return locals()[method_name]
        

def gen_embed_exceptions(input_seq, exception_types):
    try:
        for item in input_seq:
            yield item
    except exception_types as e:
        yield e
        return
        

def gen_suppress_exceptions(input_seq, exception_types):
    try:
        for item in input_seq:
            yield item
    except exception_types:
        return


"""
def c_to_mandel_journey_OLD(c):
    z = 0+0j
    while True:
        yield z
        z = z**2 + c
"""

        
def c_to_mandel_journey_abberated_by_addition(c, abberation_seq):
    raise NotImplementedError("should probably be moved to the source strings.")
    z = 0+0j
    yield z
    
    for abber in abberation_seq:
        z = z**2 + c
        z += abber
        yield z
    while True:
        z = z**2 + c
        yield z
        
        
"""
def c_must_be_in_mandelbrot(c):
    circles = [(complex(-1.0, 0.0), 0.24), (complex(0.5, 0.0),2.45)]
    for circle in circles:
        if abs(c - circle[0]) < circle[1]:
            return True
    return False
"""
     
"""
def gen_constrain_journey(journey, iter_limit, escape_radius):
    # assert escape_radius <= 256, "this may break seg intersections."
    for i, point in enumerate(journey):
        yield point
        if i >= iter_limit:
            return
        if abs(point) > escape_radius:
            return
    assert False, "incomplete journey."
"""
















def get_sum_of_inverse_segment_lengths(constrained_journey):
    result = 0.0
    for previousPoint, point in gen_track_previous(constrained_journey):
        if previousPoint is not None:
            currentSegLen = abs(point-previousPoint)
            try:
                result += 1.0 / currentSegLen
            except ZeroDivisionError:
                result = math.inf
    # print("returning {}.".format(result))
    return result
    
    
def get_sum_of_inverse_abs_vals(constrained_journey):
    result = 0.0
    for point in constrained_journey:
        currentVal = abs(point)
        try:
            result += 1.0 / currentVal
        except ZeroDivisionError:
            result = math.inf
    return result
    
    
def count_float_local_minima(input_seq): # does not recognize any minimum with more than one identical value in a row.
    raise NotImplementedError("tests needed! and maybe rewrite using gen_track_recent...")
    result = 0
    history = [None, None]
    for item in input_seq:
        if None not in history:
            if history[1] < history[0] and history[1] < item:
                result += 1
        history[0] = history[1]
        history[1] = item
    return result
                
                

    







    
def gen_seg_seq_intersections_with_seg(seg_seq, reference_seg, intersection_fun=None):
    for oldSeg in seg_seq:
        intersection = intersection_fun(reference_seg, oldSeg)
        if intersection is not None:
            yield intersection


def gen_seg_seq_self_intersections(seg_seq, intersection_fun=None, preloaded_seg_history=None, freeze_seg_history=None, gap_size=None, sort_by_time=None, combine_colinear=None):
    assert sort_by_time is not None
    assert freeze_seg_history is not None
    if sort_by_time:
        assert combine_colinear is not None
    else:
        assert not combine_colinear, "bad args"
        
    if preloaded_seg_history is None:
        segHistory = []
        assert freeze_seg_history is False
    else:
        segHistory = preloaded_seg_history
        assert freeze_seg_history, "are you sure? if so, remove this assertion to use this feature."
    
    for currentSeg in seg_seq:
        if not freeze_seg_history:
            segHistory.append(currentSeg)
        intersectionGen = gen_seg_seq_intersections_with_seg(segHistory[:-1-gap_size], currentSeg, intersection_fun=intersection_fun)
        if sort_by_time:
            intersectionList = sorted(intersectionGen, key=(lambda point: abs(currentSeg[0]-point))) # WOW, that looks slow.
            if len(intersectionList) > 0:
                if combine_colinear:
                    yield intersectionList[0]
                    if len(intersectionList) > 1:
                        yield intersectionList[-1]
                else:
                    for intersection in intersectionList:
                        yield intersection
        else:
            for intersection in intersectionGen:
                yield intersection


def gen_path_self_intersections(journey, intersection_fun=None, sort_by_time=None, combine_colinear=False): #could use less memory.
    return gen_seg_seq_self_intersections(gen_track_previous_full(journey, allow_waste=True), intersection_fun=intersection_fun, freeze_seg_history=False, gap_size=1, sort_by_time=sort_by_time, combine_colinear=combine_colinear)
    
assert_equal(list(gen_path_self_intersections([complex(0,0),complex(0,4),complex(2,2),complex(-2,2), complex(-2,3),complex(10,3)], intersection_fun=SegmentGeometry.segment_intersection, sort_by_time=False)), [complex(0,2), complex(0,3),complex(1,3)])


def gen_path_pair_mutual_intersections(point_seq_0, point_seq_1, intersection_fun=None):
    segList0 = list(gen_track_previous_full(point_seq_0, allow_waste=True))
    segGen1 = gen_track_previous_full(point_seq_1, allow_waste=True)
    # let the gap size be 0 because regardless of gap size, segs are never actually compared to their predecessor in this method!
    return gen_seg_seq_self_intersections(segGen1, intersection_fun=intersection_fun, preloaded_seg_history=segList0, freeze_seg_history=True, gap_size=0, sort_by_time=False, combine_colinear=False) 

assert_equal(list(gen_path_pair_mutual_intersections([1+2j, 3+2j, 30+2j, 30+3j, 29+3j, 29+1j], [2+1j, 2+3j, 2+30j, 3+30j, 3+29j, 1+29j], intersection_fun=SegmentGeometry.segment_intersection)), [2+2j])


# print("tests needed for path pair windowed mutual intersections.")
def gen_path_pair_windowed_mutual_intersections(point_seq_0, point_seq_1, intersection_fun=None, window_distance=None, skip_count=0):
    raise NotImplementedError("tests needed! also, verify usage of izip_shortest is correct.")
    assert window_distance >= 1
    assert 0 <= skip_count < window_distance # this window distance test I'm not so sure about.
    segGenPair = [gen_track_previous_full(pointSeq, allow_waste=True) for pointSeq in (point_seq_0, point_seq_1)]
    segWindowGenPair = [gen_track_recent_trimmed(segGen, count=window_distance+1) for segGen in segGenPair]
    for leftWindow, rightWindow in izip_shortest(*segWindowGenPair):
        leftOnRightGen = (intersection_fun(leftWindow[0], otherSeg) for otherSeg in rightWindow[skip_count:])
        rightOnLeftGen = (intersection_fun(rightWindow[0], otherSeg) for otherSeg in leftWindow[max(skip_count, 1):])
        intersectionGen = (item for item in itertools.chain(leftOnRightGen, rightOnLeftGen) if item is not None)
        for intersection in intersectionGen:
            yield intersection
        














"""
def gen_path_self_non_intersections(journey, intersection_fun=None): # code duplication, but there's no other fast way.
    knownSegs = []
    for currentSeg in gen_track_previous_full(journey):
        knownSegs.append(currentSeg)
        for oldKnownSeg in knownSegs[:-2]:
            if intersection_fun(currentSeg, oldKnownSeg) is not None:
                break
        else:
            yield currentSeg[1]
"""

# disabled because it is probably better to zip them elsewhere to avoid confusion.
"""
def gen_ladder_rung_self_intersections(journey0, journey1, intersection_fun=None):
    return gen_seg_seq_self_intersections(izip(journey0, journey1), intersection_fun=intersection_fun)
"""
                    
def gen_path_intersections_with_seg(journey, reference_seg, intersection_fun=None):
    raise NotImplementedError("possibly redefine using call to seg seq methods, and create new tests.")
    for currentSeg in gen_track_previous_full(journey):
        intersection = intersection_fun(currentSeg, reference_seg)
        if intersection is not None:
            yield intersection


def gen_path_zipped_multi_seg_intersections(journey, reference_segs, intersection_fun=None):
    for currentSeg in gen_track_previous_full(journey):
        intersections = [intersection_fun(currentSeg, referenceSeg) for referenceSeg in reference_segs]
        if any(intersection is not None for intersection in intersections):
            yield intersections

"""
def gen_path_pair_mutual_intersections(journies, intersection_fun=None):
    assert len(journies) == 2
    knownSegsByJourney = [[] for i in range(len(journies))]
    for currentSegs in zip(gen_track_previous_full(
"""


def gen_record_breakers(input_seq, score_fun=None):
    try:
        first, inputGen = take_first_and_iter(input_seq)
    except ProvisionError:
        return
    record = score_fun(first)
    yield first
    for inputItem in inputGen:
        score = score_fun(inputItem)
        if score > record:
            record = score
            yield inputItem


def gen_flag_multi_record_breakers(input_seq, score_funs=None):
    try:
        first, inputGen = take_first_and_iter(input_seq)
    except ProvisionError:
        return
    records = [scoreFun(first) for scoreFun in score_funs]
    yield (first, [True for i in range(len(score_funs))])
    for inputItem in inputGen:
        scores = [scoreFun(inputItem) for scoreFun in score_funs]
        newRecordFlags = tuple((score > record) for score, record in zip(scores, records))
        if any(newRecordFlags):
            for i, (score, isNewRecord) in enumerate(zip(scores, newRecordFlags)):
                if isNewRecord:
                    records[i] = score
            yield (inputItem, newRecordFlags)
            
"""
        if not any((score > record) for score, record in zip(scores, records)):
            continue
        currentResult = 
        for i, (score, record) in enumerate(zip(scores, records)):
            if score > record:
                records[i] = score
                currentResult[i] = item
        yield currentResult
"""
    
def gen_track_sum(input_seq):
    try:
        sumSoFar, inputGen = take_first_and_iter(input_seq)
    except ProvisionError:
        return
    yield (sumSoFar, sumSoFar)
    for item in inputGen:
        sumSoFar += item
        yield (sumSoFar, item)
assert_equal(list(gen_track_sum([1,2,3,4.5])), [(1,1),(3,2),(6,3),(10.5,4.5)])
    
    
def gen_track_mean(input_seq):
    for denominator, (sumSoFar, item) in enumerate(gen_track_sum(input_seq), 1):
        yield (sumSoFar/float(denominator), item)
assert_equal(list(gen_track_mean([1,2,3,2])), [(1.0,1),(1.5,2),(2.0,3),(2.0,2)])
assert_equal(list(gen_track_mean([complex(4,40),complex(0,0)])), [(complex(4,40), complex(4,40)), (complex(2,20), complex(0,0))])
           

def gen_track_decaying_mean(input_seq, feedback=None):
    feedbackCompliment = 1.0-feedback
    
    try:
        first, inputGen = take_first_and_iter(input_seq)
    except ProvisionError:
        return
    memoryValue = feedbackCompliment*first
    yield (memoryValue, first)
    for item in inputGen:
        memoryValue = (feedback*memoryValue) + (feedbackCompliment*item)
        yield (memoryValue, item)
        
        
def gen_change_basis_using_embedded_triplets(input_seq):
    left, middle, right = (None, None, None)
    for i, (left, middle, right) in enumerate(gen_track_recent(input_seq, count=3, default=0j)):
        if i == 0:
            continue
        yield middle.real*left + middle.imag*right
    if right is not None:
        yield right.real*middle
assert_equal(list(gen_change_basis_using_embedded_triplets([1+2j, 20+30j, 11+12j])), [2*(20+30j), 20*(1+2j)+30*(11+12j), 11*(20+30j)])
assert_equal(list(gen_change_basis_using_embedded_triplets([1+2j, 3+4j])), [2*(3+4j), 3*(1+2j)])
assert_equal(list(gen_change_basis_using_embedded_triplets([1+2j])), [0+0j])


def gen_change_basis_using_zipped_triplets(input_seq):
    raise NotImplementedError()


class SetMathProvisionError(Exception):
    pass


def mean(input_seq):
    sumSoFar = 0
    i = -1
    for i, item in enumerate(input_seq):
        sumSoFar += item
    itemCount = i + 1
    if itemCount == 0:
        raise SetMathProvisionError("This used to return 0 here. Is that allowed?")
        # return 0
    assert itemCount > 0
    return sumSoFar / float(itemCount)
    
assert mean([3,4,5]) == 4
assert mean([1,1,1,5]) == 2
assert mean([1,2]) == 1.5


def median(input_seq):
    inputList = sorted(input_seq)
    if len(inputList) == 0:
        raise SetMathProvisionError()    
    centerIndex = len(inputList)//2
    if len(inputList) % 2 == 1:
        return inputList[centerIndex]
    else:
        assert len(inputList) % 2 == 0
        return (inputList[centerIndex]+inputList[centerIndex-1])/2.0
        
assert median([1,2,3,50,400,500,600]) == 50




def complex_decomposed_median(input_seq):
    inputList = [item for item in input_seq]
    realMedian = median(reals_of(inputList))
    imagMedian = median(imags_of(inputList))
    return complex(realMedian, imagMedian)
    
assert complex_decomposed_median([1+600j,4+500j,5+55j,9+400j,125+43j,126+44j,127+45j]) == 9+55j



def farcancel_median(input_seq, _enumerateToDepthTwoPacked=(lambda thing: enumerate_to_depth_packed(thing, depth=2))):
    inputList = [item for item in input_seq]
    
    if len(inputList) == 0:
        raise SetMathProvisionError()
    if len(inputList) == 1:
        return inputList[0]
    if len(inputList) == 2:
        return mean(inputList)
    
    distances = [[abs(itemA-itemB) for itemB in inputList] for itemA in inputList]
    descendingDistanceSegs = sorted(_enumerateToDepthTwoPacked(distances), key=(lambda thing: -thing[1]))
    demirroredDescendingDistanceSegs = [item for item in descendingDistanceSegs if item[0][0] < item[0][1]]
    
    lastValidSegment = None
    cancelledIndicesSet = set()
    for segment in demirroredDescendingDistanceSegs:
        if segment[1] == math.inf:
            raise NotImplementedError("can't handle infinite distances yet!")
        if segment[0][0] in cancelledIndicesSet or segment[0][1] in cancelledIndicesSet:
            continue
        else:
            if lastValidSegment is not None:
                assert segment[1] <= lastValidSegment[1]
            lastValidSegment = segment
            for index in segment[0]:
                assert index not in cancelledIndicesSet
                cancelledIndicesSet.add(index)
                
    assert lastValidSegment is not None
    if len(cancelledIndicesSet) == len(inputList):
        lastValidSegmentEndpoints = [inputList[index] for index in lastValidSegment[0]]
        assert len(lastValidSegmentEndpoints) == 2
        assert abs(lastValidSegmentEndpoints[1] - lastValidSegmentEndpoints[0]) == lastValidSegment[1]
        return mean(lastValidSegmentEndpoints)
    else:
        assert len(cancelledIndicesSet) == len(inputList)-1
        for i, point in enumerate(inputList):
            if i not in cancelledIndicesSet:
                return point
        assert False, "failed somehow."
    assert False
    
assert_equal(farcancel_median([1+1j,3+3j,7+7j,5+5j,4+4j,2+2j,6+6j]), 4+4j)
assert_equal(farcancel_median([100+100j,300+300j,100+105j,100+95j,-200-200j,110+110j,90+90j]), 100+100j)
assert_equal(farcancel_median([0+1j,0+0j, complex(100,100)]), 0+1j)
        
        
        
def gen_linear_downsample(input_seq, count=None, analysis_fun=None):
    """
    inputGen = iter(input_seq)
    while True:
        currentBucket = [item for item in itertools.islice(inputGen, 0, count)]
        if len(currentBucket) == 0:
            return
        yield analysis_fun(currentBucket)
        if len(currentBucket) < count:
            return
    assert False
    """
    for chunk in gen_chunks_as_lists(input_seq, count):
        yield analysis_fun(chunk)

assert_equal(list(gen_linear_downsample([1,3,2,4,3,5,10,20], count=2, analysis_fun=mean)), [2,3,4,15])






def gen_shrinking_selections_as_lists(input_seq):
    inputList = [item for item in input_seq]
    combinationTupleListGen = (list(itertools.combinations(inputList, size)) for size in range(len(inputList),0,-1))
    selectionTupleGen = itertools.chain.from_iterable(combinationTupleListGen)
    return selectionTupleGen

assert list(gen_shrinking_selections_as_lists(range(0,3))) == [(0,1,2),(0,1),(0,2),(1,2),(0,),(1,),(2,)]


def gen_shrinking_selection_analyses(input_seq, analysis_fun=None):
    return (analysis_fun(selection) for selection in gen_shrinking_selections_as_lists(input_seq))













def gen_path_seg_lerps(input_seq, t=None):
    raise NotImplementedError("tests needed!")
    assert 0.0 <= t <= 1.0
    for pointA, pointB in gen_track_previous_full(input_seq):
        yield lerp(pointA, pointB, t)
        
"""
def gen_path_seg_multi_lerps(input_seq, t_seq):
    # assert isinstance(t_list, (tuple, list))
    return itertools.chain.from_iterable(izip(gen_path_seg_lerps(input_seq, t) for t in t_seq))
"""

def gen_path_seg_multi_lerps(input_seq, t_list=None): # could easily be faster with a multi lerp method.
    raise NotImplementedError("tests needed!")
    assert isinstance(t_list, (tuple, list))
    # assert all(0.0 <= t <= 1.0 for t in t_list)
    for pointA, pointB in gen_track_previous_full(input_seq):
        for t in t_list:
            yield lerp(pointA, pointB, t)
            
# gen_path_seg_multi_lerps_12x = SegmentGeometry.compose_single_arg_function(gen_path_seg_multi_lerps, depth=12)
# gen_path_seg_quarterbevel_12x = (lambda input_seq: gen_path_seg_multi_lerps(input_seq, t_list=[0.25, 0.75]))
    
"""
def gen_path_seg_midpoints(input_seq):
    return gen_path_seg_lerps(input_seq, t=0.5)
"""



def make_list_copier_from_list_mutator(input_mutator):
    def inner(input_seq, **kwargs):
        workingList = [item for item in input_seq]
        input_mutator(workingList)
        return workingList
    return inner


def sort_with_greedy_neighbor_distance_minimizer(input_list, distance_fun=None):
    for i in range(len(input_list)-1):
        bestNextItemRelIndex, bestNextItem = find_left_min(distance_fun(input_list[i], item) for item in input_list[i+1:])
        # assert input_list[i+1:][bestNextItemRelIndex] == bestNextItem
        bestNextItemIndex = bestNextItemRelIndex + i + 1
        # assert distance_fun(input_list[i], input_list[bestNextItemIndex]) == bestNextItem, (bestNextItem, i, bestNextItemRelIndex, input_list)
        if bestNextItemIndex != i + 1:
            input_list[i + 1], input_list[bestNextItemIndex] = (input_list[bestNextItemIndex], input_list[i+1])
            
            
def sort_to_greedy_shortest_path_order(input_list):
    return sort_with_greedy_neighbor_distance_minimizer(input_list, (lambda testValA, testValB: abs(testValA - testValB)))
    
testList = [complex(1,1),complex(3,1),complex(2,5),complex(2,2)]
sort_to_greedy_shortest_path_order(testList)
assert_equal(testList, [complex(1,1),complex(2,2),complex(3,1),complex(2,5)])
del testList


def sort_to_greedy_longest_path_order(input_list):
    return sort_with_greedy_neighbor_distance_minimizer(input_list, (lambda testValA, testValB: -abs(testValA - testValB)))
    
testList = [complex(1,1),complex(3,1),complex(2,5),complex(2,2)]
sort_to_greedy_longest_path_order(testList)
assert_equal(testList, [complex(1,1),complex(2,5),complex(3,1),complex(2,2)])
del testList
    
    
sorted_to_greedy_shortest_path_order = make_list_copier_from_list_mutator(sort_to_greedy_shortest_path_order)

sorted_to_greedy_longest_path_order = make_list_copier_from_list_mutator(sort_to_greedy_longest_path_order)

def eat_in_greedy_shortest_path_order(input_list):
    raise NotImplementedError()
    



"""
def sorted_to_greedy_shortest_path_order(input_seq):
    workingList = [item for item in input_seq]
    sort_to_greedy_shortest_path_order(workingList)
    return workingList
"""









def parallel_div_complex_by_floats(view_size, screen_size):
    return complex(view_size.real/screen_size[0], view_size.imag/screen_size[1])
    
    
def parallel_mul_complex_by_floats(complex_val, float_pair):
    return complex(complex_val.real * float_pair[0], complex_val.imag * float_pair[1])
    
    
def parallel_div_complex_by_complex(val0, val1):
    return complex(val0.real/val1.real, val0.imag/val1.imag)


    
    
def ordify(string):
    return (ord(char) for char in string)    


def scaled_size(input_size, input_scale):
    assert len(input_size) == 2
    assert isinstance(input_scale, int)
    return (input_size[0]*input_scale, input_size[1]*input_scale)
    
    
"""
class CoordinateError(Exception):
    pass
"""
    
"""
class ExtremeScaleWarning(Exception):
    pass
"""
"""
class ViewOutOfBoundsError(Exception):
    pass
"""

class ViewBaseCoordinateError(Exception):
    pass
    
class ViewOutOfStrictBoundsError(ViewBaseCoordinateError):
    pass
    
class ViewOutOfInttupBoundsError(ViewBaseCoordinateError):
    pass
    
class ViewOutOfMatrixBoundsError(ViewBaseCoordinateError):
    pass


def relativecpx_is_in_bounds(value):
    return value.real >= 0 and value.real < 1 and value.imag >= 0 and value.imag < 1
    
def inttup_is_in_bounds(int_tup, size):
    return not (int_tup[0] < 0 or int_tup[0] >= size[0] or int_tup[1] < 0 or int_tup[1] >= size[1])

def tunnel_absolutecpx(value, view0, view1, bound=True, default=ViewOutOfStrictBoundsError):
    return view1.relativecpx_to_absolutecpx(view0.absolutecpx_to_relativecpx(value, bound=bound, default=default), bound=bound, default=default)

def gen_tunnel_absolutecpx(input_seq, *args, **kwargs):
    assert "default" not in kwargs
    for item in input_seq:
        if item == COMPLEX_NAN:
            print_and_reduce_repetition("gen_tunnel_absolutecpx: warning: complex nan was already in the data. it will disappear.")
        result = tunnel_absolutecpx(item, *args, **kwargs, default=COMPLEX_NAN)
        if result == COMPLEX_NAN:
            continue
        yield result


class View:
    def __init__(self, *, center_pos=None, corner_pos=None, sizer=None):
        self.sizer = sizer
        assert self.sizer.real > 0
        assert self.sizer.imag > 0
        
        if corner_pos is not None:
            self.corner_pos = corner_pos
            assert center_pos is None
        else:
            assert center_pos is not None
            self.corner_pos = center_pos - 0.5*self.sizer
            
            
    @property
    def center_pos(self):
        return self.corner_pos + 0.5*self.sizer
        
        
    def relativecpx_to_absolutecpx(self, value, *, bound=True, default=ViewOutOfStrictBoundsError):
        if bound:
            if not relativecpx_is_in_bounds(value):
                if isinstance(default, type):
                    raise default()
                else:
                    return default
        return self.corner_pos + complex(self.sizer.real*value.real, self.sizer.imag*value.imag)
        
    def absolutecpx_to_relativecpx(self, value, bound=True, default=ViewOutOfStrictBoundsError):
        positionAdjusted = value - self.corner_pos
        result = complex(positionAdjusted.real/self.sizer.real, positionAdjusted.imag/self.sizer.imag)
        if bound:
            if not relativecpx_is_in_bounds(result):
                if isinstance(default, type):
                    #print("raising a {}.".format(default))
                    raise default()
                    
                else:
                    #print("returning {}.".format(default))
                    return default
        return result
        
        
    def absolutecpx_is_in_bounds(self, value):
        return self.relativecpx_is_in_bounds(self.absolutecpx_to_relativecpx(value))
    
    
    def tunnel_absolutecpx_to(self, value, other_view, *, bound=True, default=ViewOutOfStrictBoundsError):
        return other_view.relativecpx_to_absolutecpx(self.absolutecpx_to_relativecpx(value, bound=bound, default=default), bound=bound, default=default)
        
    def tunnel_absolutecpx_from(self, value, other_view, *, bound=True, default=ViewOutOfStrictBoundsError):
        return self.relativecpx_to_absolutecpx(other_view.absolutecpx_to_relativecpx(value, bound=bound, default=default), bound=bound, default=default)


    def relativecpx_to_inttup(self, value, size):
        """
        if size[0] < 2 or size[1] < 2:
            raise ExtremeScaleWarning()
        """
        result = (math.floor(value.real*size[0]), math.floor(value.imag*size[1]))
        if not inttup_is_in_bounds(result, size):
            raise ViewOutOfInttupBoundsError()
        return result
        
    def inttup_to_relativecpx(self, int_tup, size):
        result = complex(float(int_tup[0])/size[0], float(int_tup[1])/size[1])
        if not relativecpx_is_in_bounds(result):
            raise ViewOutOfInttupBoundsError()
        return result


    def absolutecpx_to_inttup(self, value, size):
        return self.relativecpx_to_inttup(self.absolutecpx_to_relativecpx(value), size)
        
    def inttup_to_absolutecpx(self, int_tup, size):
        return self.relativecpx_to_absolutecpx(self.inttup_to_relativecpx(int_tup, size))
        
        
    def absolutecpx_to_matrix_cell(self, value, *, matrix=None, size=None):
        assert len(matrix) == size[1]
        assert len(matrix[0]) == size[0]
        try:
            x, y = self.absolutecpx_to_inttup(value, size)
        except OverflowError:
            raise ViewOutOfMatrixBoundsError("near-infinity can never be a list index!")
        if x < 0 or y < 0:
            raise ViewOutOfMatrixBoundsError("negatives not allowed here.")
        if y >= len(matrix):
            raise ViewOutOfMatrixBoundsError("y is too high.")
        row = matrix[y]
        if x >= len(row):
            raise ViewOutOfMatrixBoundsError("x is too high.")
        return row[x]


    def gen_cell_descriptions(self, size, reverse_x=False):
        for x, y in higher_range(corners_to_range_descriptions(stop_corner=size), iteration_order=(0,1), post_slices=[slice(None,None,-1), None] if reverse_x else None):
            yield (x, y, self.inttup_to_absolutecpx((x, y), size))
        
        
    def get_sub_view_sizer(self, size):
        return complex(self.sizer.real/size[0], self.sizer.imag/size[1])
        
        
    def gen_sub_view_descriptions(self, size):
        if size[0]*size[1] > 16384:
            print("View.gen_sub_views: warning: {} ({}x{}) sub views is a lot.".format(size[0]*size[1], size[0], size[1]))
        subViewSizer = self.get_sub_view_sizer(size)
        for x, y, subViewCornerAbsolutecpx in self.gen_cell_descriptions(size):
            yield (x, y, View(corner_pos=subViewCornerAbsolutecpx, sizer=subViewSizer))





class GridSettings:
    def __init__(self, view, grid_size):
        assert all(is_round_binary(item) for item in grid_size)
        self.grid_size = tuple(iter(grid_size)) # make it a tuple as a standard for equality tests in other places.
        self.view = view
        assert self.view.sizer.real > 0.0
        assert self.view.sizer.imag > 0.0
    
    def get_cell_sizer(self):
        return self.view.get_sub_view_sizer(self.grid_size)
        
    def get_cell_width(self):
        return self.get_cell_sizer().real
        
    def gen_cell_descriptions(self):
        return self.view.gen_cell_descriptions(self.grid_size)

    """
    def absolutecpx_to_matrix_cell(self, value, matrix=None):
        return self.view.absolutecpx_to_matrix_cell(value, matrix=matrix, size=self.grid_size)
    """



def bumped(data, amount):
    return [item+amount for item in data]
    


class Camera:
    def __init__(self, view, screen_size=None, bidirectional_supersampling=None):
        self.view, self.screen_size, self.bidirectional_supersampling = (view, screen_size, bidirectional_supersampling)
        supersize = scaled_size(self.screen_size, self.bidirectional_supersampling)
        self.seed_settings = GridSettings(self.view, supersize)
        self.screen_settings = GridSettings(self.view, self.screen_size)
        
        

        



class Echo:
    def __init__(self, length=None, default=None):
        assert length > 0, "length must be defined and greater than 0."
        self.length = length
        self.history = collections.deque([default for i in range(self.length)])
        
    def push(self, item):
        self.history.append(item)
        while len(self.history) > self.length:
            self.history.popleft()
            
    @property
    def current(self):
        return self.history[-1]
        
    @current.setter
    def current(self, new_value):
        self.push(new_value)
        
    @property
    def previous(self):
        return self.history[-2]


def vec_add_vec(vec0, vec1):
    for i, vec1val in enumerate(vec1):
        vec0[i] += vec1val
        
        
def vec_add_scaled_vec(vec0, vec1, scale=None):
    for i, vec1val in enumerate(vec1):
        vec0[i] += vec1val * scale

    
def vec_add_vec_masked(vec0, vec1, mask):
    for i in range(len(vec0)):
        if mask[i]:
            vec0[i] += vec1[i]
        

def vec_add_scalar_masked(vec0, input_scalar, mask):
    for i in range(len(vec0)):
        if mask[i]:
            vec0[i] += input_scalar
            



"""
def gen_drop_first_if_equals(input_seq, value):
    inputGen = iter(input_seq)
    first = next(inputGen)
"""



def gen_floats_after_fxp_ca(input_float_seq, steps=1, ca_nonabox_stepper=None):
    fxpTemplate = fxpmath.Fxp(0.0, signed=True, n_word=64, n_int=32)
    bitListSeq = ([int(bitChar) for bitChar in fxpmath.Fxp(currentFloat, like=fxpTemplate).bin()] for currentFloat in input_float_seq)
    for i in range(steps):
        bitListSeq = CGOL.ca_gen_stepped_rows(bitListSeq, ca_nonabox_stepper=ca_nonabox_stepper, x_edge_mode=CGOL.EdgeMode.VOID, y_edge_mode=CGOL.EdgeMode.SHRINK)
    
    resultGen = ((fxpmath.Fxp("0b"+"".join(str(bitInt) for bitInt in bitList), like=fxpTemplate)).__float__() for bitList in bitListSeq)
    return resultGen














# modifiedJourney = gen_recordbreakers(journeySelfIntersections, score_fun=abs)
# modifiedJourney = (point-seed for point in constrainedJourney[1:])
# recordBreakersJourney = gen_multi_recordbreakers(constrainedJourney[1:], score_funs=[abs, inv_abs_of, (lambda inputVal: get_complex_angle(ensure_nonzero(inputVal)))])

#gspFromOriginJourney, constrainedJourney = (constrainedJourney, None); sort_to_greedy_shortest_path_order(gspFromOriginJourney); assert gspFromOriginJourney[0] == 0
#gspFromOriginJourneySelfIntersections = gen_path_self_intersections(gspFromOriginJourney, intersection_fun=SegmentGeometry.segment_intersection)

# sortedByAbsJourney, constrainedJourney = (constrainedJourney, None); list.sort(sortedByAbsJourney, key=abs)
# sortedByAbsJourneySelfIntersections = gen_path_self_intersections(sortedByAbsJourney, intersection_fun=SegmentGeometry.segment_intersection)

# sortedBySeedmanhdistJourney, constrainedJourney = (constrainedJourney[1:], None); list.sort(sortedBySeedmanhdistJourney, key=(lambda pt: SegmentGeometry.complex_manhattan_distance(seed, pt))); assert sortedBySeedmanhdistJourney[0]==seed
# sortedBySeedmanhdistJourneySelfIntersections = gen_path_self_intersections(sortedBySeedmanhdistJourney, intersection_fun=SegmentGeometry.segment_intersection)

# shuffledJourney, constrainedJourney = (constrainedJourney, None); random.shuffle(shuffledJourney)

# exponentiatedJourney = MatrixMath.matrix_exp(MatrixMath.make_square_matrix_from_column(constrainedJourney)).T[0].flatten().tolist()[0]
"""
zjfiJourneyToFollow = sortedByAbsJourneySelfIntersections # skip first item here if necessary.
zjfiJourneyToAnalyze = sortedByAbsJourneySelfIntersections
# zjfiFoundationSegsToUse = [(complex(0,0), seed), (seed, zjfiJourneyToAnalyze[-1]), (complex(0,0), zjfiJourneyToAnalyze[-1])]  assert escape_radius==2.0
zjfiFoundationSegsToUse = [(seed, max(escape_radius,2.0)*10.0*get_normalized(seed)), (complex(0,seed.imag), seed), (complex(seed.real,0), seed)]; assert camera.view.size.real < 10, "does a new spoke length for foundation tests need to be chosen?"
zippedJourneyFoundationIntersections = gen_path_zipped_multi_seg_intersections(zjfiJourneyToFollow, reference_segs=zjfiFoundationSegsToUse, intersection_fun=SegmentGeometry.segment_intersection); # assert len(zjfiJourneyToAnalyze) < iter_limit, "bad settings! is this a buddhabrot, or is it incorrectly an anti-buddhabrot or a joint-buddhabrot?"; assert zjfiJourneyToAnalyze[0] == complex(0,0), "what? bad code?";
"""

# differential mode:
"""
if i == 0:
    print("in differential mode, the first point's journey is not drawn.")
else:
    comparisonMaskedVisitPointPairList = [pointPair for pointPair in zip(visitPointListEcho.previous, visitPointListEcho.current) if abs(pointPair[1]-pointPair[0]) < 1*pixelWidth]

    for ii, (leftPoint, centerPoint) in enumerate(comparisonMaskedVisitPointPairList):
        assert ii <= point_limit
        drawingStats["dotCount"] += 1
        drawPoint(mainPoint=centerPoint, comparisonPoint=leftPoint)
    # modify_visit_count_matrix(visitCountMatrix, ((curTrackPt, True, (curTrackPt.real>prevTrackPt.real), (curTrackPt.imag>prevTrackPt.imag)...
"""

def mark_if_true(thing, name):
    if thing:
        return str(thing)+name
    else:
        return ""
        
def get_virtual_2d_index(index=None, *, size=None):
    assert 0 <= index[1] < size[1]
    assert 0 <= index[0] < size[0]
    return index[1]*size[0] + index[0]
    
def gen_ljusted(input_seq, length, default=None, crop=False):
    yieldCount = 0
    itemGen = itertools.islice(input_seq, 0, length) if crop else iter(input_seq)
    for item in itemGen:
        yield item
        yieldCount += 1
    while yieldCount < length:
        yield default
        yieldCount += 1
    


@measure_time_nicknamed("do_buddhabrot", end="\n\n", include_lap=True)
def do_buddhabrot(dest_surface, camera, iter_skip=None, iter_limit=None, point_skip=None, point_limit=None, count_scale=1, fractal_formula=None, esc_exceptions=None, buddha_type=None, banded=None, do_top_half_only=False, skip_origin=False, w_int=None, w_step=None, custom_window_size=None, do_visual_test=False):
    SET_LIVE_STATUS("started...")
    print("do_buddhabrot started.")
    assert None not in (iter_skip, iter_limit, point_skip, point_limit, esc_exceptions, buddha_type, banded)
    if not camera.screen_settings.grid_size == dest_surface.get_size():
        print("warning: screen_settings.grid_size and dest surface size are not equal!")
    if w_int is not None or w_step is not None:
        assert False, "not defined right now!"
        # w = w_step*w_int
        # w_compliment = 1.0-w
    if custom_window_size is not None:
        assert False, "not defined right now!"
        
    # top(RallGincrvsleftBincivsleft)bottom(up)      # polarcross(RseedouterspokeGseedhorizlegBseedvertleg)  # journeyAndDecaying(0.5feedback)MeanSeqLadderRungPolarCross
    # test_bb_8xquarterbevel                         # greedyShortPathFromSeed_rectcross_RallGincrBinci      # _sortedBySeedmanhdist
    # (wf(z,normz)-wf(0,normc))_(w={}*{})_rectcross  # pathDownsampCpxDecompMedian_windowWidth{}_polarcross  # _draw(top(path)bottom(home))
    # (path_ver_plus_home_ver)                       # fxpCA(nonaboxMax)                                     # journeyWrapToMatRows_exp_unwrapRows
    # rowOfJournsToMatCols_fill(1+1j)_exp
    
    setSummaryStr = "{}(ini({})yld({})esc({})itr({}))_repeatJournToMatCols_sin_getRowSums_RallGincrBinci".format(buddha_type, fractal_formula["init_formula"], fractal_formula["yield_formula"], fractal_formula["esc_test"], fractal_formula["iter_formula"], custom_window_size)
    viewSummaryStr = "{}pos{}fov{}{}itrLim{}{}ptLim{}biSup{}count".format(camera.view.center_pos, camera.view.sizer, mark_if_true(iter_skip,"itrSkp"), iter_limit, mark_if_true(point_skip,"ptSkp"), point_limit, camera.bidirectional_supersampling, count_scale)
    output_name = to_portable("{}_{}_{}_".format(setSummaryStr, viewSummaryStr, COLOR_SETTINGS_SUMMARY_STR))
    print("output name is {}.".format(repr(output_name)))
    
    escstopJourneyFun = compile_mandel_method("c_to_escstop_mandel_journey", **fractal_formula)
    
    
    pixelWidth = camera.screen_settings.get_cell_width()
    if not pixelWidth < 0.1:
        print("Wow, are pixels supposed to be that small?")
    
    visitCountMatrix = construct_data(camera.screen_settings.grid_size[::-1], default_value=[0,0,0])
    homeOutputMatrix = construct_data(camera.screen_settings.grid_size[::-1], default_value=[0,0,0])
    
    
    def _specializedDraw():
        draw_squished_ints_to_surface(dest_surface, visitCountMatrix+homeOutputMatrix, access_order="yxc")
    def specializedDrawAndSave(namePrefix=None):
        _specializedDraw()
        save_surface_as(dest_surface, name_prefix=namePrefix)
        pygame.display.flip()
    
    
    
    def pointToMatrixCell(matrix, point):
        return camera.screen_settings.view.absolutecpx_to_matrix_cell(point, matrix=matrix, size=camera.screen_settings.grid_size)
        
    """
    def drawPointUsingMask(matrix, mainPoint=None, mask=None):
        try:
            vec_add_scalar_masked(pointToMatrixCell(matrix, mainPoint), count_scale, mask)
        except CoordinateError:
            pass
    """
    print("maybe catching ViewBaseCoordinateError is too wide a net.")
    def drawPointUsingNumMask(matrix, *, mainPoint=None, mask=None, draw_scale=1):
        try:
            vec_add_scaled_vec(pointToMatrixCell(matrix, mainPoint), mask, scale=count_scale*draw_scale)
        except ViewBaseCoordinateError:
            pass
            
    def drawZippedPointTupleToChannels(matrix, *, mainPoints=None, draw_scale=1):
        assert len(mainPoints) == 3, "what?"
        for i, currentPoint in enumerate(mainPoints):
            if currentPoint is None:
                continue
            try:
                currentCell = pointToMatrixCell(matrix, currentPoint)
                currentCell[i] += count_scale*draw_scale
            except CoordinateError:
                pass
                
    def drawPointUsingComparison(matrix, *, mainPoint=None, comparisonPoint=None, draw_scale=1):
        drawPointUsingNumMask(matrix, mainPoint=mainPoint, mask=[True, mainPoint.real>comparisonPoint.real, mainPoint.imag>comparisonPoint.imag], draw_scale=draw_scale)
    
    """
    def drawPointUsingManyComparisons(matrix, mainPoint=None, comparisonPoints=None):
        finalNumMask = [sum(currentTuple[i] for currentTuple in ([True, mainPoint.real>comparisonPoint.real, mainPoint.imag>comparisonPoint.imag] for comparisonPoint...
    """
    
    
    
    def genCellDescriptions(testMode=False, **other_kwargs):
        if testMode:
            # return [(point[0], point[1], camera.seed_settings.whole_to_complex(point, centered=False)) for point in test_seeds]
            raise NotImplementedError()
        else:
            return camera.seed_settings.gen_cell_descriptions(**other_kwargs)
            
            
    def seedToPointList(seed):
        _constrainedJourneyGen = gen_suppress_exceptions(itertools.islice(escstopJourneyFun(seed), iter_skip, iter_limit+1), esc_exceptions)
        constrainedJourney = list(_constrainedJourneyGen)
        journeyUnskippedLen = len(constrainedJourney) + iter_skip
        
        if buddha_type == "jbb":
            seedIsInSet = True
        else:
            seedEscapes = not (journeyUnskippedLen >= iter_limit)
            if buddha_type == "bb":
                seedIsInSet = seedEscapes
            elif buddha_type == "abb":
                seedIsInSet = not seedEscapes
            else:
                assert False, "invalid buddha type {}.".format(buddha_type)
        
        if (not seedIsInSet):
            return []
        else:
            normedSeed = get_normalized(seed, undefined_result=0j)
            if skip_origin:
                assert normedSeed != 0j
                assert seed != 0j
                
            # normedJourneyMinusNormedC = gen_suppress_exceptions(((w_compliment*item+w*get_normalized(item, undefined_result=0j))-(w_compliment*0+w*normedSeed) for item in constrainedJourney), (ZeroDivisionError,))
            
            # normedJourneyMinusNormedCStagedSelfIntersections = normedJourneyMinusNormedC
            # normedJourneyMinusNormedCStagedSelfIntersections = gen_path_self_intersections(normedJourneyMinusNormedCStagedSelfIntersections, intersection_fun=SegmentGeometry.segment_intersection, sort_by_time=False, combine_colinear=False)
            
            # normedJourneyStagedSelfIntersections = gen_path_self_intersections(normedJourneyStagedSelfIntersections, intersection_fun=SegmentGeometry.rect_seg_polar_space_intersection, sort_by_time=False, combine_colinear=False)
            
            # pointGens = [gen_path_pair_windowed_mutual_intersections(constrainedJourney, constrainedJourney, intersection_fun=SegmentGeometry.segment_intersection, window_distance=windowDist, skip_count=2) for windowDist in (5,13,29)]
            # pointGens = [gen_linear_downsample_using_mean(constrainedJourney, count=windowSize) for windowSize in (17,19,23)]
            # zippedPointGen = izip_longest(*pointGens)
            
            # downsampledPathPointGen = gen_linear_downsample(constrainedJourney, count=custom_window_size, analysis_fun=complex_decomposed_median)
            # downsampledPathSelfIntersectionGen = gen_path_self_intersections(downsampledPathPointGen, intersection_fun=SegmentGeometry.rect_seg_polar_space_intersection, sort_by_time=False)
            
            #journeyStagedSelfIntersectionGen = constrainedJourney
            # for iii in range(1):
            #journeyStagedSelfIntersectionGen = gen_path_self_intersections(journeyStagedSelfIntersectionGen, intersection_fun=SegmentGeometry.segment_intersection, sort_by_time=False)
            # journeyStagedSelfIntersectionGen = gen_path_self_intersections(journeyStagedSelfIntersectionGen, intersection_fun=SegmentGeometry.rect_seg_polar_space_intersection, sort_by_time=False)
            
            # modifiedPointGen = gen_shrinking_selection_analyses(constrainedJourney, analysis_fun=mean)
            # modifiedPointGen = (complex(realPart,imagPart) for realPart, imagPart in zip(*[gen_floats_after_fxp_ca(curSeq, ca_nonabox_stepper=CGOL.nonabox_max) for curSeq in (reals_of(constrainedJourney), imags_of(constrainedJourney))]))
            # modifiedPathSelfIntersectionGen = gen_path_self_intersections(modifiedPointGen, intersection_fun=SegmentGeometry.segment_intersection, sort_by_time=True)
            
            # journeyWithTrackedDecayingMean = gen_track_decaying_mean(constrainedJourney, feedback=0.5)
            # journeyAndDecayingMeanSeqLadderRungSelfIntersections = gen_seg_seq_self_intersections(journeyWithTrackedDecayingMean, intersection_fun=SegmentGeometry.segment_intersection)
            # journeySelfNonIntersections = gen_path_self_non_intersections(constrainedJourney, intersection_fun=SegmentGeometry.segment_intersection)
            
            # limitedVisitPointGen = gen_suppress_exceptions(itertools.islice(constrainedJourney, point_skip, point_limit), (ProvisionError,))
            # return list(limitedVisitPointGen)
            
            
            #journeyMat = MatrixMath.repeat_to_square_matrix_columns(constrainedJourney)
            #modifiedJourneyMat = MatrixMath.matrix_sin(journeyMat)
            #return MatrixMath.matrix_rows_flattened_to_list(modifiedJourneyMat)
            #return MatrixMath.get_row_sums(...)
            return constrainedJourney
            """
            # exponentiatedJourneyMat = MatrixMath.matrix_exp(journeyMat)
            try:
                invertedJourneyMat = MatrixMAth.matrix_inv(journeyMat)
            except MatrixMath.SingularMatrixInversionError as smie: # singular matrix
                return []
            # flatMat = MatrixMath.matrix_rows_flattened_to_list(exponentiatedJourneyMat)
            return list(MatrixMath.gen_matrix_rows(invertedJourneyMat))
            """
            
        assert False
            
    
    def drawPointList(seed, pointList, *, draw_scale=1):
        for ii, currentItem in enumerate(pointList):
            assert ii <= point_limit, "bad configuration."
            # drawZippedPointTupleToChannels(visitCountMatrix, currentItem)
            # drawPointUsingMask(visitCountMatrix, mainPoint=currentItem[0], mask=currentItem[1])
            #if currentItem.imag < 0:
            assert isinstance(currentItem, complex), type(currentItem)
            drawPointUsingComparison(visitCountMatrix, mainPoint=currentItem, comparisonPoint=seed, draw_scale=draw_scale)
            #if seed.imag > 0:
            drawPointUsingComparison(homeOutputMatrix, mainPoint=seed, comparisonPoint=currentItem, draw_scale=draw_scale)
    
    # def drawPointLists
    
    subSideSize = 1
    inRowMode = False
    
    screenSubViews = list(subView for _, _, subView in camera.screen_settings.view.gen_sub_view_descriptions((subSideSize, subSideSize)))
    
    visitPointListEcho = Echo(length=(camera.seed_settings.grid_size[0] if inRowMode else 2))
    
    print("done initializing.")
    
    
    for x, y, seed in genCellDescriptions(testMode=False):
        
        if (x==0):
            if do_top_half_only and (y >= camera.seed_settings.grid_size[1]//2):
                break
            if banded and (y%(camera.seed_settings.grid_size[1]//IMAGE_BAND_COUNT) == 0):
                SET_LIVE_STATUS("drawing and saving...")
                specializedDrawAndSave(namePrefix=output_name+"{}of{}rows_".format(y, camera.seed_settings.grid_size[1]))
                SET_LIVE_STATUS("saved...")
            if CAPTION_RATE_LIMITER.get_judgement():
                SET_LIVE_STATUS("{}of{}rows".format(y, camera.seed_settings.grid_size[1]))
        
        if skip_origin and (seed == 0j):
            pointListForSeed = []
        else:
            pointListForSeed = seedToPointList(seed)
        visitPointListEcho.push(pointListForSeed)
        
        if inRowMode:
            assert False
        
            assert not skip_origin
            
            if x == camera.seed_settings.grid_size[0]-1:
                assert_equal(point_limit - point_skip, camera.seed_settings.grid_size[0])
                assert_equal(camera.seed_settings.grid_size[0], len(visitPointListEcho.history))
                # for item in visitPointListEcho.history:
                #    assert isinstance(item, list), item
                # (abs(point + complex(pointListX,pointIndex)) for pointIndex,point in enumerate(pointList)
                currentMat = MatrixMath.numpy.matrix([list(gen_ljusted(pointList, len(visitPointListEcho.history), default=complex(1+1j), crop=True)) for pointListX, pointList in enumerate(visitPointListEcho.history)]).T
                exponentiatedMat = MatrixMath.matrix_exp(currentMat, dtype=complex)
                
                #print(exponentiatedMat)
                # print("mat done")
                """
                try:
                    invertedMat = MatrixMath.get_normalized_matrix(MatrixMath.matrix_inv(currentMat))*4
                except MatrixMath.SingularMatrixInversionError:
                    print("singular matrix of shape {}.".format(currentMat.shape))
                    continue
                """
                assert False
                    
                for pointListX, pointList in enumerate(MatrixMath.gen_matrix_columns(exponentiatedMat)):
                    pointListSeed = camera.screen_settings.view.inttup_to_absolutecpx((pointListX, y), camera.seed_settings.grid_size)
                    assert pointListSeed.imag == seed.imag
                    drawPointList(pointListSeed, pointList)
            
        else:
            assert not inRowMode
            # if (x==0):
            #    visitPointListEcho.push([])
            
            if subSideSize == 1:
                drawPointList(seed, visitPointListEcho.current)
            else:
                assert False, "not a good design yet!"
                """
                for period in range(1, subSide+1):
                    for offset in range(0, period):
                        subView = screenSubViews[get_virtual_2d_index((period-1, offset), size=(subSide, subSide))]
                        subViewSeed = tunnel_absolutecpx(seed, camera.screen_settings.view, subView, bound=False)
                        subViewPointList =  list(gen_tunnel_absolutecpx(visitPointListEcho.current[offset::period], camera.screen_settings.view, subView, bound=True))
                        drawPointList(subViewSeed, subViewPointList, draw_scale=period)
                """
                """
                for rowIndex, subView in enumerate(screenSubViews):
                    if rowIndex >= len(visitPointListEcho.current):
                        break
                    subViewSeed = tunnel_absolutecpx(seed, camera.screen_settings.view, subView, bound=False)
                    subViewPointList = list(gen_tunnel_absolutecpx(visitPointListEcho.current[rowIndex], camera.screen_settings.view, subView, bound=True))
                    drawPointList(subViewSeed, subViewPointList)
                """
        
        
    print("doing final draw and save...")
    specializedDrawAndSave(namePrefix=output_name)
    print("do_buddhabrot done.")
    SET_LIVE_STATUS("done.")













def quadrilateral_is_convex(points):
    assert len(points) == 4
    return SegmentGeometry.segments_intersect((points[0], points[2]), (points[1], points[3]))




def check_bb_containedness(argless_itercount_fun=None, iter_limit=None, buddhabrot_set_type=None):
    if buddhabrot_set_type in {"bb", "abb"}:
        itercount = argless_itercount_fun()
        escapesBeforeIterLimit = itercount is not None
        isInDesiredSet = True
        if buddhabrot_set_type == "bb":
            if not escapesBeforeIterLimit:
                isInDesiredSet = False
        else:
            assert buddhabrot_set_type == "abb"
            if escapesBeforeIterLimit:
                isInDesiredSet = False

    else:
        assert buddhabrot_set_type == "jbb"
        isInDesiredSet = True
    return isInDesiredSet



def gen_ordered_index_pairs(start, stop):
    for Ai in range(start, stop-1):
        for Bi in range(Ai+1, stop):
            yield (Ai, Bi)
            
def gen_ordered_item_pairs(input_list):
    for Ai, Bi in gen_ordered_index_pairs(0, len(input_list)):
        yield (input_list[Ai], input_list[Bi])


"""
PanelCell = collections.NamedTuple("PanelCellName", ["seed", "current_z", "previous_z", "set_membership"])
testPanelCell = PanelCell(0+0j,1+1j,2+2j,True)
assert testPanelCell.seed == 0+0j
del testPanelCell
"""
i_SEED, i_PREVIOUS_Z, i_CURRENT_Z, i_ISINSET = (0, 1, 2, 3)


@measure_time_nicknamed("create_panel")
def create_panel(seed_settings, iter_limit=None, escape_radius=None, buddhabrot_set_type=None, centered_sample=None, kill_not_in_set=None):
    print("constructing empty panel...")

    assert kill_not_in_set is not None
    assert kill_not_in_set, "are you sure?"
    assert seed_settings.grid_size[0] <= 4096, "make sure there is enough memory for this!"
    panel = construct_data(seed_settings.grid_size[::-1], default_value=None)
    cToMandelItercountFast = compile_mandel_method("c_to_mandel_itercount_fast", init_formula="z=0j", yield_formula="z", esc_test="abs(z)>{}".format(escape_radius), iter_formula="z=z**2+c")
    arglessItercountFun = (lambda: cToMandelItercountFast(seed, iter_limit))
    
    print("populating panel...")
    
    assert abs(seed_settings.graveyard_point) > escape_radius
    for x, y, seed in seed_settings.iter_cell_descriptions(centered=centered_sample):
        panelCell = [seed, 0.0+0.0J, 0.0+0.0J, None]
        isInSet = check_bb_containedness(argless_itercount_fun=arglessItercountFun, iter_limit=iter_limit, buddhabrot_set_type=buddhabrot_set_type)
        panelCell[i_ISINSET] = isInSet
        if (not isInSet) and kill_not_in_set:
            panelCell[i_CURRENT_Z], panelCell[i_PREVIOUS_Z] = (None, None)
        panel[y][x] = panelCell
        if x == 0 and STATUS_RATE_LIMITER.get_judgement():
            print("create_panel: {}%...".format(str(int(float(100*y)/seed_settings.grid_size[1])).rjust(2," ")))
            
    yes, no = (0.0, 0.0)
    for panelCell in iterate_to_depth(panel, depth=2):
        if panelCell[i_ISINSET]:
            yes += 1.0
        else:
            no += 1.0
    print("{}% of cells are in the set.".format(round((yes/(yes+no))*100.0, ndigits=3)))
            
    print("done creating panel.")
    return panel






def get_four_neighbor_items(matrix, x, y):
    if not x > 0 and y > 0:
        raise IndexError("no.")
    return [matrix[neighborY][neighborX] for neighborX,neighborY in ((x, y-1), (x+1,y), (x,y+1), (-x, y))]


"""
class Dummies(Enum):
    NOT_TRACKED = "not_tracked"
"""



        

NotTracked = summon_cactus("NotTracked")

def clamp_positive(val):
    return val if val >= 0 else 0
    
    
def split_to_lists(input_seq, trigger_fun=None, include_empty=True):
    inputGen = iter(input_seq)
    currentList = []
    while True:
        try:
            currentItem = next(inputGen)
        except StopIteration:
            if include_empty or len(currentList) > 0:
                yield currentList
            return
        if trigger_fun(currentItem):
            if include_empty or len(currentList) > 0:
                yield currentList
            currentList = []
        else:
            currentList.append(currentItem)
    assert False
    
    
    
    
    
    




    
    

@measure_time_nicknamed("do_panel_buddhabrot", end="\n\n", include_lap=True)
def do_panel_buddhabrot(camera, iter_limit=None, output_iter_limit=None, output_interval_iters=1, blank_on_output=True, count_scale=1, escape_radius=None, buddhabrot_set_type="bb", headstart=None, skip_zero_iter_image=False, kill_not_in_set=True, w_int=None, w_step=None):
    assert buddhabrot_set_type in {"bb", "jbb", "abb"}
    assert buddhabrot_set_type == "bb", "is it really ready?"
    assert kill_not_in_set, "are you sure you're ready?"
    assert iter_limit is not None
    if output_iter_limit is None:
        output_iter_limit = iter_limit
        print("output_iter_limit defaulting to value of iter_limit, which is {}.".format(iter_limit))
    if w_int is not None: # duplicate of nonpanel code, do not touch.
        assert False, "it's not prepared."
        # w = w_step*w_int
        # w_compliment = 1.0-w
    # assert blank_on_output, "not ready yet? also remove this assertion in loop body if ready, to avoid a disappointing crash after panel init!"
    
    # outputColorSummary = "R012outofsetneighG3outofsetneighB4outofsetneigh"
    # outputColorSummary = "top(RguestpaircmidptbothinsetGoneinsetBneitherinset)bottom(endpt)"
    # neighborPathSimultaneousCross
    # 4NeighborZShareChooseNearestToC 4NeighborZShareChooseRandom
    # zpos(RdisttocGrBi)
    # 2linsub_agletized
    # pow(wf(z,normz),wf(1,normc))_(w={}*{})
    # _post(pow(normz,normc))
    # adjMutuRectcross_windowdistance(R5G10C15)
    setSummaryStr = "panel{}_{}_splitlessSlice_gradMean_rectcross_ReithrGrowCcol".format("(headst({}))".format(headstart) if headstart is not None else "", buddhabrot_set_type) # , w_step, w_int)
    viewSummaryStr = "{}pos{}fov{}esc{}itrlim{}biSup{}count_{}".format(camera.view.center_pos, camera.view.size, escape_radius, iter_limit, camera.bidirectional_supersampling, count_scale, ("blank" if blank_on_output else "noBlank"))
    output_name = to_portable("{}_{}_{}_".format(setSummaryStr, viewSummaryStr, COLOR_SETTINGS_SUMMARY_STR))
    
    print("creating visitCountMatrix...")
    
    screenSpaceHomeMatrix, visitCountMatrix = (summon_cactus("screenSpaceHomeMatrix_is_disabled."), construct_data(camera.screen_settings.grid_size[::-1], default_value=[count_scale*0, count_scale*0, count_scale*0]), )
    # assert tuple(shape_of(visitCountMatrix)) == camera.screen_settings.grid_size[::-1]+(3,)
    
    
    def pointToVisitCountMatrixCell(point): # DUPLICATE CODE  DO NOT MODIFY
        return camera.screen_settings.complex_to_item(visitCountMatrix, point, centered=False)
    def drawPointUsingMask(mainPoint=None, mask=None): # DUPLICATE CODE  DO NOT MODIFY
        # drawingStats["dotCount"] += 1
        try:
            vec_add_scalar_masked(pointToVisitCountMatrixCell(mainPoint), count_scale, mask)
            # drawingStats["drawnDotCount"] += 1
        except CoordinateError:
            pass # drawnDotCount won't be increased.
    def drawPointUsingComparison(mainPoint=None, comparisonPoint=None): # DUPLICATE CODE  DO NOT MODIFY
        drawPointUsingMask(mainPoint=mainPoint, mask=[True, mainPoint.real>comparisonPoint.real, mainPoint.imag>comparisonPoint.imag])
        
    def drawPointSeqUsingMask(inputSeq, mask=None):
        for point in inputSeq:
            drawPointUsingMask(mainPoint=point, mask=mask)
   
    def drawZippedPointSeq(inputSeq):
        for item in inputSeq:
            for i in range(3):
                if item[i] is not None:
                    try:
                        pointToVisitCountMatrixCell(item[i])[i] += count_scale
                    except CoordinateError:
                        pass
    
    
    
    def specializedDraw(inputMatrix):
        draw_squished_ints_to_screen(inputMatrix, access_order="yxc")
    def specializedDrawAndScreenshot(inputMatrix, name_prefix=None):
        assert name_prefix is not None
        specializedDraw(inputMatrix)
        save_screenshot_as(name_prefix=name_prefix)
    def blankIfNeeded(inputMatrix):
        if blank_on_output:
            fill_data(inputMatrix, count_scale*0)
    
    panel = create_panel(camera.seed_settings, iter_limit=iter_limit, escape_radius=escape_radius, buddhabrot_set_type=buddhabrot_set_type, centered_sample=False, kill_not_in_set=kill_not_in_set)
    
    """
    hotelGrid = construct_data(camera.seed_settings.grid_size[::-1], default_value=[])
    assert hotelGrid[0][0] is not hotelGrid[0][1]
    print("done creating hotelGrid.")
    """
    
    def iteratePanelCell(panelCell):
        if panelCell[i_CURRENT_Z] is not None:
            if kill_not_in_set:
                assert panelCell[i_ISINSET] is True
            panelCell[i_PREVIOUS_Z], panelCell[i_CURRENT_Z] = (panelCell[i_CURRENT_Z], panelCell[i_CURRENT_Z]**2 + panelCell[i_SEED])
            if abs(panelCell[i_CURRENT_Z]) > escape_radius:
                panelCell[i_PREVIOUS_Z], panelCell[i_CURRENT_Z] = (None, None)
    
    
    # HEADSTART
    if headstart is None:
        print("no headstart to give.")
    else:
        print("giving headstart of {}...".format(headstart))
        _startNotNone = sum(panelCell[i_CURRENT_Z] is not None for panelCell in iterate_to_depth(panel, depth=2))
        for (y, x), panelCell in enumerate_to_depth_packed(panel, depth=2):
            if x == 0:
                if STATUS_RATE_LIMITER.get_judgement():
                    print("giving headstart: ~{}%...".format(int(y*100.0/len(panel))))
            localHeadstart = eval(headstart)
            assert localHeadstart >= 0 and isinstance(localHeadstart, int)
            for i in range(localHeadstart):
                iteratePanelCell(panelCell)
        _endNotNone = sum(panelCell[i_CURRENT_Z] is not None for panelCell in iterate_to_depth(panel, depth=2))
        print("done giving headstart. {}% of z values that started non-None remain non-None.".format(round(100*_endNotNone/float(_startNotNone), ndigits=3)))
    
    
    
    for iter_index in range(0, output_iter_limit):
    
        # SHOW
        SET_LIVE_STATUS("n={}".format(iter_index))
        if ((iter_index % output_interval_iters) == 0) and not (iter_index == 0 and skip_zero_iter_image):
            specializedDrawAndScreenshot(visitCountMatrix, name_prefix=output_name+"{}of{}itrs_".format(iter_index, iter_limit))
            blankIfNeeded(visitCountMatrix)
            # assert blank_on_output, "not ready yet?"
    
        # ITERATE
        for (y, x), panelCell in enumerate_to_depth_packed(panel, depth=2):
            iteratePanelCell(panelCell)
        """
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            if panelCell[i_CURRENT_Z] is not None:
                try:
                    neighboringActivePanelCellZVals = [panelCell[i_PREVIOUS_Z]] + [item[i_PREVIOUS_Z] for item in get_four_neighbor_items(panel, x, y) if item[i_PREVIOUS_Z] is not None]
                except IndexError:
                    continue
                # indexClosestToHome, _ = find_left_min(abs(val-panelCell[i_SEED]) for val in neighboringActivePanelCellZVals)
                
                panelCell[i_CURRENT_Z] = random.choice(neighboringActivePanelCellZVals)
        """
        
                    
        # DRAW
        """
        v256ovrScrDiag = 256.0/abs(camera.view.size)
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            if y in (0, len(panel)-1) or x in (0, len(panel[0])-1):
                continue
            if panelCell[i_CURRENT_Z] is None:
                continue
            if not panelCell[i_ISINSET]:
                continue
            # neighboringPanelCells = get_four_neighbor_items(panel, x, y)
            # drawPointUsingComparison(mainPoint=panelCell[i_CURRENT_Z], comparisonPoint=panelCell[i_SEED])
            itemToEdit = camera.screen_settings.complex_to_item(screenSpaceHomeMatrix, panelCell[i_SEED], centered=False)
            
            itemToEdit[0], itemToEdit[1], itemToEdit[2] = (abs(panelCell[i_CURRENT_Z]-panelCell[i_SEED])*v256ovrScrDiag, clamp_positive(panelCell[i_CURRENT_Z].real*v256ovrScrDiag), clamp_positive(panelCell[i_CURRENT_Z].imag*v256ovrScrDiag))
        """
        """
            # neighborPathSimultaneousCross:
            for neighboringPanelCell in neighboringPanelCells:
                if neighboringPanelCell[i_CURRENT_Z] is None:
                    continue
                intersectionPoint = SegmentGeometry.segment_intersection((panelCell[i_PREVIOUS_Z], panelCell[i_CURRENT_Z]), (neighboringPanelCell[i_PREVIOUS_Z], neighboringPanelCell[i_CURRENT_Z]))
                if intersectionPoint is not None:
                    drawPointUsingComparison(mainPoint=intersectionPoint, comparisonPoint=panelCell[i_SEED])
        """
        
        
        def drawFromCellSeqInLoop(inputCellSeq, mask=None, include_unprocessed=None):
            assert include_unprocessed is False, "not ready"
            # ptSplitListGen = split_to_lists(inputSeq, trigger_fun=(lambda thing: thing is None), include_empty=False)
            # agletizedPtSplitListGen = (ptList[:-1][::2]+[ptList[-1]] for ptList in ptSplitListGen)
            # ptSeq = itertools.chain.from_iterable(agletizedPtSplitListGen)
            splitlessPointGen = (cell[i_CURRENT_Z] for cell in inputCellSeq if cell[i_CURRENT_Z] is not None)
            gradualMeanGen = (item[0] for item in gen_track_mean(splitlessPointGen))
            gradualMeanSelfIntersectionGen = gen_path_self_intersections(gradualMeanGen, intersection_fun=SegmentGeometry.segment_intersection, sort_by_time=False)
            
            # zcPairGen = ((cell[i_CURRENT_Z], cell[i_SEED]) for cell in inputSeq if cell[i_CURRENT_Z] is not None and cell[i_CURRENT_Z] != 0j)
            # moddedPointGen = ((w_compliment*zcPair[0]+w*get_normalized(zcPair[0], undefined_result=0j))**(w_compliment*1+w*get_normalized(zcPair[1], undefined_result=0j)) for zcPair in zcPairGen)
            
            # normPointSeq = [get_normalized(item, undefined_result=0) for item in pointSeq]
            # moddedPointSeqSelfIntersections = gen_path_self_intersections(moddedPointGen, intersection_fun=SegmentGeometry.segment_intersection, sort_by_time=False)
            # pointSeqDoubleSelfIntersections = gen_path_self_intersections(pointSeqSelfIntersections, intersection_fun=SegmentGeometry.segment_intersection)
            # newBasisPointSeq = gen_change_basis_using_embedded_triplets(pointSeq)
            drawPointSeqUsingMask(gradualMeanSelfIntersectionGen, mask=mask)
            # if include_unprocessed:
            #    drawPointSeqUsingMask(, mask=[True,False,False])
                
        for y in range(len(panel)):
            cellRow = [panel[y][x] for x in range(len(panel[y]))] # could be faster.
            drawFromCellSeqInLoop(cellRow, mask=[True, True, False], include_unprocessed=False)
        for x in range(len(panel[0])):
            cellCol = [panel[y][x] for y in range(len(panel))]
            drawFromCellSeqInLoop(cellCol, mask=[True, False, True], include_unprocessed=False)
        
        """
        def drawFromAdjacentCellSeqsInLoop(cellSeq0, cellSeq1):
            pointSeq0, pointSeq1 = [[cell[i_CURRENT_Z] for cell in cellSeq if cell[i_CURRENT_Z] is not None] for cellSeq in (cellSeq0, cellSeq1)]
            mutualIntersectionsGens = [gen_path_pair_windowed_mutual_intersections(pointSeq0, pointSeq1, intersection_fun=SegmentGeometry.segment_intersection, window_distance=windowDist) for windowDist in (5, 10, 15)]
            zippedPointsGen = izip_longest(*mutualIntersectionsGens)
            drawZippedPointSeq(zippedPointsGen)
            
        for y in range(1, len(panel)):
            cellRow0, cellRow1 = [[panel[yy][x] for x in range(len(panel[0]))] for yy in (y-1, y)]
            drawFromAdjacentCellSeqsInLoop(cellRow0, cellRow1)
        for x in range(1, len(panel[0])):
            cellCol0, cellCol1 = [[panel[y][xx] for y in range(len(panel))] for xx in (x-1, x)]
            drawFromAdjacentCellSeqsInLoop(cellCol0, cellCol1)
        """
            
        
        # hotel code:
        """
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            try:
                hotel = camera.seed_settings.complex_to_item(hotelGrid, panelCell[i_CURRENT_Z], centered=False)
            except CoordinateError:
                continue # out of bounds
            hotel.append(panelCell[i_SEED])
            
        for hotelY, hotelX, hotel in enumerate_to_depth(hotelGrid, depth=2):
            if len(hotel) < 2:
                continue
                
            for guestA, guestB in gen_ordered_item_pairs(hotel):
                guestPairMidpoint = (guestA + guestB) / 2.0
                
                guestApanelCell, guestBpanelCell = [camera.seed_settings.complex_to_item(panel, guest, centered=False) for guest in (guestA, guestB)]
                guestAisInSet, guestBisInSet = (guestApanelCell[i_ISINSET], guestBpanelCell[i_ISINSET])
                mask = [(guestAisInSet == guestBisInSet == True), (guestAisInSet != guestBisInSet), (guestAisInSet == guestBisInSet == False)]
                
                for itemIsAllowed, item in [(guestA.imag > 0, guestA), (guestB.imag > 0, guestB), (guestPairMidpoint.imag < 0, guestPairMidpoint)]:
                    if not itemIsAllowed:
                        continue
                    try:
                        visitCountMatrixCell = camera.screen_settings.complex_to_item(visitCountMatrix, item, centered_sample=False)
                    except CoordinateError:
                        continue # the point is not on the screen.
                    vec_add_scalar_masked(visitCountMatrixCell, count_scale, mask)
            
        for hotelY, hotelX, hotel in enumerate_to_depth(hotelGrid, depth=2):
            hotel.clear()
        """
                
    specializedDrawAndScreenshot(visitCountMatrix, name_prefix=output_name)
    blankIfNeeded(visitCountMatrix)
                
                
                
                
def panel_brot_draw_close_encounters():
    raise NotImplementedError("?")


def panel_brot_draw_panel_based_on_neighbors_in_set(seed_settings=None, panel=None, visit_count_matrix=None, count_scale=None, centered_sample=None):
    for y, panelRow in itertools.islice(enumerate(panel), 1, len(panel)-1):
        for x, panelCell in itertools.islice(enumerate(panelRow), 1, len(panelRow)-1):
            if not panelCell[i_ISINSET]: # if this point isn't in the desired set:
                continue

            try:
                visitCountMatrixCell = seed_settings.complex_to_item(visit_count_matrix, panelCell[i_CURRENT_Z], centered_sample=centered_sample)
            except CoordinateError:
                continue # the point is not on the screen.

            # assert abs(panelCellZ) < escape_radius, (x, y, panelCellZ, abs(panelCellZ), escape_radius) # can't always be true because the point could have just escaped.

            cellNeighborhood = [panel[y-1][x], panel[y][x+1], panel[y+1][x], panel[y][x-1]]
            #neighborhoodEscapeCount = sum((abs(testCell[1]) > escape_radius) for testCell in cellNeighborhood)
            neighborhoodOutOfSetCount = sum((not testCell[i_ISINSET]) for testCell in cellNeighborhood)
            #neighborhoodIsConvex = quadrilateral_is_convex(pointNeighborhood)
            #neighborHasEscaped = max(abs(point) for point in pointNeighborhood) > escape_radius # THIS IS ALSO TRUE WHEN the point is in a graveyard (not part of desired set).
            #westNeighborZ = panelRow[x-1][1]
            #eastNeighborZ = panelRow[x+1][1]
            #southNeighborZ = panel[y-1][x][1]
            #northNeighborZ = panel[y+1][x][1]
            #horizNoEscapedNeighbor = (abs(westNeighborZ) < escape_radius and abs(eastNeighborZ) < escape_radius)
            #vertNoEscapedNeighbor = (abs(southNeighborZ) < escape_radius and abs(northNeighborZ) < escape_radius)
            if neighborhoodOutOfSetCount < 3:
                visitCountMatrixCell[0] += count_scale
            #if neighborHasEscaped:
            #if horizNoEscapedNeighbor and abs(panelCellZ-westNeighborZ) > abs(panelCellZ-eastNeighborZ):
            if neighborhoodOutOfSetCount == 3:
                visitCountMatrixCell[1] += count_scale
            #if not neighborhoodIsConvex:
            #if vertNoEscapedNeighbor and abs(panelCellZ-southNeighborZ) > abs(panelCellZ-northNeighborZ):
            if neighborhoodOutOfSetCount == 4:
                visitCountMatrixCell[2] += count_scale



"""
class Canvas:
    def __init__(self, size):
        self.size = size
        assert all(item in {4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384} for item in self.size), "are you sure about that?"
    def draw_squished_ints(self, data):
        raise NotImplementedError()
    def save(self):
        raise NotImplementedError()
"""

"""
def save_screenshot_as(*args, **kwargs):
    pygame.display.flip()
    save_surface_as(screen, *args, **kwargs)
    pygame.display.flip()
    
def draw_squished_ints_to_screen(*args, **kwargs):
    pygame.display.flip()
    draw_squished_ints_to_surface(screen, *args, **kwargs)
    pygame.display.flip()
"""

def SET_LIVE_STATUS(status_text):
    try:
        pygame.display.set_caption("z. " + status_text + " " + OUTPUT_FOLDER)
    except Exception as e:
        print("couldn't set caption to: {}. error: {}.".format(repr(status_text), e))
    if PASSIVE_DISPLAY_FLIP_RATE_LIMITER.get_judgement():
        pygame.display.flip()


if __name__ == "__main__":
    pygame.init()
    pygame.display.init()

    RASTER_SIZE = (256, 256)
    _screen = pygame.display.set_mode((RASTER_SIZE[0], 2*RASTER_SIZE[1]))

    COLOR_SETTINGS_SUMMARY_STR = "color(atan)"
    OUTPUT_FOLDER = "oS_test/256x/"


    SET_LIVE_STATUS("loading...")
    IMAGE_BAND_COUNT = (4 if _screen.get_size()[1] <= 128 else (16 if _screen.get_size()[1] <= 512 else 32))
    # assert screen.get_size()[0] == screen.get_size()[1], "are you sure about that?"





@measure_time_nicknamed("main")
def main():

    #test_abberation([0], 0, 16384)
    # for biSup, iterLim, ptLim in [(1024)]
    # "((abs(c*0.25)+0.015625)**-1)"
    # (1.75+0.5*sin(iter_index))
    # complex({}*sin(z.imag+c.real), tan(z.real+c.imag))+c
    # for steppedVal in [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0]:
    # for steppedVal in ComplexGeometry.float_range(1, 8, 0.03125):
    # z=z**2+((-1)**n)*c
    # init_formula="z,zd,zs,w=c,c,c,1", yield_formula="z.real*zd+z.imag*(zs*(w**-1))", esc_test="abs(z)>16", iter_formula="z=z*z+c;zd,zs,w=0.5*zd+0.5*z,zs+z,w+1",
    # init_formula="z,nc=c,norm(c)*0.0125*{}".format(v), yield_formula="z", esc_test="abs(z)>16", iter_formula="z=z*z+nc",
    # init_formula="z=c", yield_formula="z", esc_test="abs(z)>16", iter_formula="z=z*z+c",
    # for v in range(0, 161):
    # -1.75+0.0j, 0.5+0.03125j
    #iter_skip=0, iter_limit=24*subs, point_skip=0, point_limit=192, count_scale=8, fractal_formula={"init_formula":"z=c", "yield_formula":"for tf in dv1range({}): yield z**(2**tf)+tf*c".format(subs), "esc_test":"abs(z)>16", "iter_formula":"z=z*z+c"}
    """
    wStepCount = 128 # 256 is good
    for wInt in range(1*wStepCount+1):
    """
    #for customWindowSize in range(1,1024,8):
    # for subs in [64]: #, 1,32]:
    # center_pos=-0.14-0.86j, sizer=0.75+0.75j
    do_buddhabrot(_screen, Camera(View(center_pos=0.0j, sizer=4+4j), screen_size=RASTER_SIZE, bidirectional_supersampling=1), iter_skip=0, iter_limit=64, point_skip=0, point_limit=64, count_scale=8,
        fractal_formula={"init_formula":"z=c", "yield_formula":"yield z", "esc_test":"abs(z)>16", "iter_formula":"z=z*z+c"}, esc_exceptions=(OverflowError, ZeroDivisionError), buddha_type="bb", banded=True, skip_origin=False, do_top_half_only=False) #  custom_window_size=customWindowSize) # w_int=wInt, w_step=1.0/wSteps)
    
    # do_panel_buddhabrot(Camera(View(0+0j, 4+4j), screen_size=screen.get_size(), bidirectional_supersampling=1), iter_limit=1024, output_iter_limit=1024, output_interval_iters=2, blank_on_output=False, count_scale=4, escape_radius=16.0, headstart="16", skip_zero_iter_image=False) # w_int=wInt, w_step=1.0/wStepCount)









                
print("done testing.")

if __name__ == "__main__":
    main()
    PygameDashboard.stall_pygame(preferred_exec=THIS_MODULE_EXEC)
    print("""                   # #
     ##            # #        ##  #
    #    ### ###  ## ##      #    #
    # ## # # # # # # # # # # ###  #
    #  # ### ###  ## ##   #  #     
     ##                 ##    ##  #
     
    """)



"""

todo:
  -small display surface, large save surface.
  -logging in-window.
  -draw to multiple images at different steps in simulation.
  -desired sequences:
    -lerping from rectcross to polarcross mode.
  -testing:
    -improve extra assertions for segment_intersection with point-on-line testing.
    -visual debugger for geometry.
    -replace special answer enum.Enums with something that's easier to debug.
    -replace fuzz/defuzz testing with testing that uses only fuzzing (that is, check if fun(fuzz(inputs))==fuzz(fun(inputs))).
  -fixed-point geometry calculations.
  -color cross buddhabrot based on intersection <angle|segment convergence and/or divergence>.
  
  -tools:
    -gradual median.
    -rolling <median|mean>.
    
  -path self intersection:
    -batch with bounding rectangles. A line segment intersects a rectangle IFF ((it intersects one of that rectangle's diagonals) or (it has at least one endpoint inside the rectangle)). Also compare batch counding rectangles to each other.
    -store segment presences in quadtree. subdividing the segment is not necessary to do this.
    -refraction or reflection with previous segments.
    -combine all _intersections with a new segment_ using mean.
    -find intersections but instead yield <points opposite them on loop shapes|bounded region <center of mass|inscribed circle center>>.
    -gen intersections between a path in polar space and the same path in rect space, but only intersections not analogous to a specific self-intersection of the path in either space.
    
  -nonpanel:
    -time-smooth journies and their intersections ((z^(2^tf)+tf*c) or recursively smoothed).
    -journey smoothing by <simple splines|follower with intertia|follower with smoothly changing direction>.
    
  -panel:
    -drawing:
      -.
    -simulation:
      -matrix multiplication: MZ times MZ, MZ times MC, MC times MZ.
      -gravity simulation: affect and draw <all|few> paths.
    -distortion:
      -sort rows by z real then sort cols by z imag. maybe repeat. maybe 2d bubblesort <alternating direction 1d|shifting boxes|simultaneous, find preference weights>.
      -warp all z to roughly uniform density.
    -hotels:
      -make local density a simulation input. (also do this within journey for nonpanel).
      -modify c, <<warp towards|rot around> average|gravitate> within hotel.
      -draw density local maximum or minimum hotels only.
      -draw a line between c1 and c2 whenever z1 and z2 are very close together.
      
      
  -done:
    -lap counter. improved lap timer.
    -order intersection points on a segment by time (actually distance from start of seg).
    -use c and escape point as a new coord space...
    -journey pt -> journey pt <minus|divided by> mean of journey at that time.
    -panel:
      -drawing:
        -self-intersections of path visiting all Z in each row and col.
      
  
  
observations:
  -geometry overhaul (25 Feb 2022) brought polarcross from 14x slower than rectcross to 9.6x, and disabling assertions brings it to 7x.


what current features or design goals are kinda painful to keep around?
  -separate methods do_buddhabrot and do_panel_buddhabrot
  -trying to reduce calls.
  -differential draw mode in do_buddhabrot.
  -count_scale.
  -floats in the visitCountMatrix. they prevent fast color adjustments using a list as a map.
"""