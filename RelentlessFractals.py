#!/usr/bin/python3

"""
todo:
  -testing:
    -improve extra assertions for segment_intersection with point-on-line testing.
    -visual debugger for geometry.
    -replace special answer enum.Enums with something that's easier to debug.
    -replace fuzz/defuzz testing with testing that uses only fuzzing (that is, check if fun(fuzz(inputs))==fuzz(fun(inputs))).
  -lap counter. improved lap timer.
  -order intersection points on a segment by time (actually distance from start of seg).
  -fixed-point geometry calculations.
  -time-smooth journies and their intersections ((z^(2^s)+s*c) or recursively smoothed).
  
  
observations:
  -geometry overhaul (25 Feb 2022) brought polarcross from 14x slower than rectcross to 9.6x, and disabling assertions brings it to 7x.
"""



import time
import math
import itertools
import collections
import copy
import random
import gc

import pygame

from ColorTools import atan_squish_to_byteint_unsigned_uniform_nearest

from TestingBasics import assert_equal

from ComplexGeometry import real_of, imag_of, inv_abs_of, get_complex_angle, get_normalized
import SegmentGeometry
from SegmentGeometry import find_left_min, lerp

import ComplexGeometry

from PureGenTools import gen_track_previous, peek_first_and_iter, gen_track_previous_full, higher_range

import Trig
sin, cos, tan = (Trig.sin, Trig.cos, Trig.tan) # short names for use only in compilation of mandel methods.



testZip = zip("ab","cd")
izip = (zip if (iter(testZip) is iter(testZip)) else itertools.izip)
testZip2 = izip("ab","cd")
assert (iter(testZip2) is iter(testZip2)) and (not isinstance(testZip2, list)), "can't izip?"
del testZip, testZip2





import PygameDashboard
from PygameDashboard import measure_time_nicknamed


def THIS_MODULE_EXEC(string):
    exec(string)

CAPTION_RATE_LIMITER = PygameDashboard.SimpleRateLimiter(1.0)










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
    return result
    
    

    
    
    
    
    
    

def enumerate_from_both_ends(data):
    assert hasattr(data, "__len__")
    for forwardI, item in enumerate(data):
        yield (forwardI, len(data)-forwardI-1, item)
            
assert [item for item in enumerate_from_both_ends("abc")] == [(0,2,"a"), (1,1,"b"), (2,0,"c")]
assert [item for item in enumerate_from_both_ends("abcd")] == [(0,3,"a"), (1,2,"b"), (2,1,"c"), (3,0,"d")]
            


def is_round_binary(value):
    assert value > 0
    return value == 2**(value.bit_length()-1)
    
assert is_round_binary(1)
assert is_round_binary(2)
assert not is_round_binary(3)
    
assert is_round_binary(256)
assert not is_round_binary(255)

assert is_round_binary(65536)
assert not is_round_binary(65535)
assert not is_round_binary(65537)


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
    forbiddenDict = {
        "<":"Lss", ">":"Gtr", ":":"Cln", "\"":"Dblqt", "\\":"Bkslsh", "|":"Vrtpipe", "?":"Quemrk", "*":"Astrsk", 
        "=":"Eq", "'":"Sglqt", "!":"Exclmrk", "@":"Atsgn", "#":"Poundsgn", "$":"Dlrsgn", "%":"Prcntsgn",  "^":"Caret", "&":"Amprsnd",
    }
    for forbiddenChar, replacementChar in forbiddenDict.items():
        oldPathStr = path_str
        path_str = path_str.replace(forbiddenChar, replacementChar)
        if path_str != oldPathStr:
            print("to_portable: Warning: ~{} occurrences of {} will be replaced with {} for portability.".format(oldPathStr.count(forbiddenChar), repr(forbiddenChar), repr(replacementChar)))
    return path_str


@measure_time_nicknamed("save_surface_as", end="\n\n", include_lap=True)
def save_surface_as(surface, name_prefix="", name=None):
    if name is None:
        name = "{}{}.png".format(time.monotonic(), str(surface.get_size()).replace(", ","x"))
    usedName = to_portable(OUTPUT_FOLDER + name_prefix + name)
    print("saving file {}.".format(usedName))
    pygame.image.save(surface, usedName)
    measure_time_nicknamed("garbage collection")(gc.collect)()
    #print("{} unreachable objects.".format())



    
    

    



@measure_time_nicknamed("draw_squished_ints_to_surface")
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
            

            


            
def construct_data(size, default_value=None):
    assert len(size) > 0
    if len(size) == 1:
        return [copy.deepcopy(default_value) for i in range(size[0])]
    else:
        return [construct_data(size[1:], default_value=default_value) for i in range(size[0])]

assert_equal(shape_of(construct_data([5,6,7])), [5,6,7])


def fill_data(data, fill_value):
    assert isinstance(data, list)
    for i in range(len(data)):
        if isinstance(data[i], list):
            fill_data(data[i], fill_value)
        elif isinstance(data[i], tuple):
            raise TypeError("tuples can't be processed!")
        else:
            if not type(data[i]) == type(fill_value):
                raise NotImplementedError("type can't be changed!")
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
def c_to_mandel_itercount_fast(c, iter_limit, escape_radius):
    ${init_formula}
    for iter_index in range(iter_limit):
        if ${esc_test}:
            return iter_index
        ${iter_formula}
    return None""",
    
        "c_to_escstop_mandel_journey":"""
def c_to_escstop_mandel_journey(c):
    ${init_formula}
    for iter_index in itertools.count():
        yield z
        if ${esc_test}:
            return
        ${iter_formula}""",
        
    }
    
# z0="0+0j", exponent="2"
def compile_mandel_method(method_name, init_formula=None, iter_formula=None, esc_test=None):
    sourceStr = _mandelMethodsSourceStrs[method_name]
    assert sourceStr.count("def {}(".format(method_name)) == 1, "bad source code string for name {}!".format(method_name)
    
    sourceStr = sourceStr.replace("${init_formula}", init_formula).replace("${iter_formula}", iter_formula).replace("${esc_test}", esc_test)
    
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
    except exception_types as e:
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
                result += 2.0**256
    # print("returning {}.".format(result))
    return result
    
    
def get_sum_of_inverse_abs_vals(constrained_journey):
    result = 0.0
    for point in constrained_journey:
        currentVal = abs(point)
        try:
            result += 1.0 / currentVal
        except ZeroDivisionError:
            pass
    return result
    
    
def count_float_local_minima(input_seq): # does not recognize any minimum with more than one identical value in a row.
    result = 0
    history = [None, None]
    for item in input_seq:
        if None not in history:
            if history[1] < history[0] and history[1] < item:
                result += 1
        history[0] = history[1]
        history[1] = item
    return result
                
                

    

    


def gen_seg_seq_self_intersections(seg_seq, intersection_fun=None, gap_size=None):
    knownSegs = []
    for currentSeg in seg_seq:
        knownSegs.append(currentSeg)
        for oldKnownSeg in knownSegs[:-1-gap_size]:
            intersection = intersection_fun(currentSeg, oldKnownSeg)
            if intersection is not None:
                yield intersection
    
def gen_path_self_intersections(journey, intersection_fun=None): #could use less memory.
    return gen_seg_seq_self_intersections(gen_track_previous_full(journey, allow_waste=True), intersection_fun=intersection_fun, gap_size=1)
    
assert_equal(list(gen_path_self_intersections([complex(0,0),complex(0,4),complex(2,2),complex(-2,2), complex(-2,3),complex(10,3)], intersection_fun=SegmentGeometry.segment_intersection)), [complex(0,2), complex(0,3),complex(1,3)])
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
        first, inputGen = peek_first_and_iter(input_seq)
    except IndexError:
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
        first, inputGen = peek_first_and_iter(input_seq)
    except IndexError:
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
        sumSoFar, inputGen = peek_first_and_iter(input_seq)
    except IndexError:
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
        first, inputGen = peek_first_and_iter(input_seq)
    except IndexError:
        return
    memoryValue = feedbackCompliment*first
    yield (memoryValue, first)
    for item in inputGen:
        memoryValue = (feedback*memoryValue) + (feedbackCompliment*item)
        yield (memoryValue, item)



def gen_path_seg_lerps(input_seq, t=None):
    assert 0.0 <= t <= 1.0
    for pointA, pointB in gen_track_previous_full(input_seq):
        yield lerp(pointA, pointB, t)
        
"""
def gen_path_seg_multi_lerps(input_seq, t_seq):
    # assert isinstance(t_list, (tuple, list))
    return itertools.chain.from_iterable(izip(gen_path_seg_lerps(input_seq, t) for t in t_seq))
"""

def gen_path_seg_multi_lerps(input_seq, t_list=None): # could easily be faster with a multi lerp method.
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


def test_abberation(text, scale, iterLimit):
    if isinstance(text, str):
        abberationSeq = [((value-64)/256.0)*scale*(1+1j) for value in ordify(text)]
    else:
        abberationSeq = text
    journeyFun = lambda c: c_to_mandel_journey_abberated_by_addition(c, itertools.cycle(abberationSeq))
    for x, y, sampleCoords in get_seeds(screen.get_size(), -0.5+0j, 4+4j, centered_sample=False):
        # screen.set_at((x,y), (x%256, (x*y)%256, (x%(y+1))%256))
        if x==0 and y%4 == 0:
            pygame.display.flip()
        # itercount = c_and_journey_fun_to_itercount(journeyFun, sampleCoords, 256, 4)
        # color = automatic_color(itercount)
        journey = journeyFun(sampleCoords)
        constrainedJourney = [item for item in constrain_journey(journey, iterLimit, 4)]
        #val0 = get_sum_of_inverse_abs_vals(constrainedJourney)
        #val1 = get_sum_of_inverse_segment_lengths(constrainedJourney)
        #valb0 = max(0,val0/(val1+1))*64
        #valb1 = max(0,val1/(val0+1))*64
        #selfIntersections = count_intersections(constrainedJourney)
        #valc0 = len(constrainedJourney)
        #color = normalize_color(
        #    (val2*2/4, val2/(1+valc0)*32/4, valc0/(val2+1)*8/4)
        #)
        color = squish_color((
            count_float_local_minima(abs(item) for item in constrainedJourney)*16,
            count_float_local_minima(item.real for item in constrainedJourney)*16,
            count_float_local_minima(item.imag for item in constrainedJourney)*16,
        ))
        screen.set_at((x, y), color)
        #color = squish_color((
        #    0,
        #    x,
        #    y,
        #))
        
        #screen.set_at(
        #    (
        #        int(squish_unsigned(len(constrainedJourney), screen.get_size()[0])),
        #        int(squish_unsigned(get_sum_of_inverse_segment_lengths(constrainedJourney), screen.get_size()[1])),
        #    ),
        #    color,
        #)
    


def scaled_size(input_size, input_scale):
    assert len(input_size) == 2
    assert isinstance(input_scale, int)
    return (input_size[0]*input_scale, input_size[1]*input_scale)
    
    
class View:
    def __init__(self, center_pos, size):
        assert isinstance(center_pos, complex)
        assert isinstance(size, complex)
        self.center_pos, self.size = (center_pos, size)
        self.corner_pos = self.center_pos - (self.size / 2.0)
        
    def get_sub_view_size(self, subdivisions_pair):
        return parallel_div_complex_by_floats(self.size, subdivisions_pair)
    
    """
    def get_sub_view_corner(self, subdivisions_pair, sub_view_coord):
        # assert all(sub_view_coord[i] <= subdivisions_pair[i] for i in (0,1))
        return self.corner_pos + parallel_mul_complex_by_floats(self.get_sub_view_size(subdivisions_pair), sub_view_coord)

    def get_sub_view_center(self, subdivisions_pair, sub_view_coord):
        return self.get_sub_view_corner(subdivisions_pair, (sub_view_coord[0] + 0.5, sub_view_coord[1] + 0.5))
    """
    

def bump(data, amount):
    return [item+amount for item in data]
    


class Camera:
    def __init__(self, view, screen_size=None, bidirectional_supersampling=None):
        self.view, self.screen_size, self.bidirectional_supersampling = (view, screen_size, bidirectional_supersampling)
        supersize = scaled_size(self.screen_size, self.bidirectional_supersampling)
        self.seed_settings = GridSettings(self.view, supersize)
        self.screen_settings = GridSettings(self.view, self.screen_size)
        
    
class GridSettings:
    
    def __init__(self, view, grid_size):
        assert all(is_round_binary(item) for item in grid_size)
        self.grid_size = tuple(iter(grid_size)) # make it a tuple as a standard for equality tests in other places.
        self.view = view
        assert self.view.size.real > 0.0
        assert self.view.size.imag > 0.0
        
        self.cell_size = view.get_sub_view_size(self.grid_size)
        self.graveyard_point = (256.5 + abs(self.view.center_pos) + abs(2.0*self.view.size)) # a complex coordinate that will never appear on camera. Make it so large that there is no doubt.
        
    def whole_to_complex(self, whole_coord, centered=None):
        assert centered is not None
        return self.view.corner_pos + parallel_mul_complex_by_floats(self.cell_size, (bump(whole_coord, 0.5) if centered else whole_coord))
        
    def complex_to_whole(self, complex_coord, centered=None):
        # centered_sample might not be logically needed for the answer to this question, depending on how the screen is defined in future versions of the program.
        complexInView = complex_coord - self.view.corner_pos
        complexOfCell = parallel_div_complex_by_complex(complexInView, self.cell_size)
        return (int(complexOfCell.real), int(complexOfCell.imag))
        
    def complex_to_item(self, data, complex_coord, centered=None):
        try:
            x, y = self.complex_to_whole(complex_coord, centered=centered)
        except OverflowError:
            raise IndexError("near-infinity can never be a list index!")
        if x < 0 or y < 0:
            raise IndexError("negatives not allowed here.")
        return data[y][x]
        
    def iter_cell_whole_coords(self, range_descriptions=None, swap_iteration_order=False):
        if range_descriptions is None:
            range_descriptions = [(0, s, 1) for s in self.grid_size]
        iterationOrder = [1,0]
        if swap_iteration_order:
            iterationOrder = iterationOrder[::-1]
        return higher_range(range_descriptions, iteration_order=iterationOrder)
        
    def iter_cell_descriptions(self, range_descriptions=None, swap_iteration_order=False, centered=None):
        assert centered is not None
        for x, y in self.iter_cell_whole_coords(range_descriptions=range_descriptions, swap_iteration_order=swap_iteration_order):
            yield (x, y, self.whole_to_complex((x,y), centered=centered))
            
testGrid = GridSettings(View(0+0j, 4+4j), (2,2))
SegmentGeometry.assert_nearly_equal(list(testGrid.iter_cell_descriptions(centered=False)), [(0, 0, -2-2j), (1, 0, 0-2j), (0, 1, -2+0j), (1, 1, 0+0j)])
SegmentGeometry.assert_nearly_equal(list(testGrid.iter_cell_descriptions(centered=True)), [(0, 0, -1-1j), (1, 0, 1-1j), (0, 1, -1+1j), (1, 1, 1+1j)])
del testGrid
        



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



@measure_time_nicknamed("do_buddhabrot")
def do_buddhabrot(camera, iter_limit=None, point_limit=None, count_scale=1, init_formula=None, iter_formula=None, esc_test=None, esc_exceptions=None, buddha_type=None, banded=True):
    SET_LIVE_STATUS("started...")
    print("do_buddhabrot started.")
    assert None not in (iter_limit, point_limit, init_formula, iter_formula, esc_test, esc_exceptions, buddha_type)
    # top(RallGincrvsleftBincivsleft)bottom(up)
    # polarcross(RseedouterspokeGseedhorizlegBseedvertleg)
    # journeyAndDecaying(0.5feedback)MeanSeqLadderRungPolarCross
    # test_bb_8xquarterbevel
    # greedyShortPathFromSeed_rectcross_RallGincrBinci
    # _sortedBySeedmanhdist
    setSummaryStr = "{}(ini({})itr({})esc({}))_RallGincrBinci".format(buddha_type, init_formula, iter_formula, esc_test)
    viewSummaryStr = "{}pos{}fov{}itrlim{}ptlim{}biSuper{}count".format(camera.view.center_pos, camera.view.size, iter_limit, point_limit, camera.bidirectional_supersampling, count_scale)
    output_name = to_portable("{}_{}_{}_".format(setSummaryStr, viewSummaryStr, COLOR_SETTINGS_SUMMARY_STR))
    assert camera.screen_settings.grid_size == screen.get_size()
    print("output name is {}.".format(repr(output_name)))
    
    escstopJourneyFun = compile_mandel_method("c_to_escstop_mandel_journey", init_formula=init_formula, iter_formula=iter_formula, esc_test=esc_test)
    
    
    def specializedDraw():
        draw_squished_ints_to_screen(visitCountMatrix, access_order="yxc")
    
    drawingStats = {"dotCount": 0, "drawnDotCount":0}
    
    pixelWidth = camera.screen_settings.cell_size.real
    if not pixelWidth < 0.1:
        print("Wow, are pixels supposed to be that small?")
    
    visitCountMatrix = construct_data(camera.screen_settings.grid_size[::-1], default_value=[0,0,0])
    
    
    def pointToVisitCountMatrixCell(point):
        return camera.screen_settings.complex_to_item(visitCountMatrix, point, centered=False)
    def drawPointUsingMask(mainPoint=None, mask=None):
        drawingStats["dotCount"] += 1
        try:
            vec_add_scalar_masked(pointToVisitCountMatrixCell(mainPoint), count_scale, mask)
            drawingStats["drawnDotCount"] += 1
        except IndexError:
            pass # drawnDotCount won't be increased.
    def drawZippedPointsToChannels(mainPoints=None):
        assert len(mainPoints) == 3, "what?"
        for i, currentPoint in enumerate(mainPoints):
            if currentPoint is None:
                continue
            drawingStats["dotCount"] += 1
            try:
                currentCell = pointToVisitCountMatrixCell(currentPoint)
                currentCell[i] += count_scale
                drawingStats["drawnDotCount"] += 1
            except IndexError:
                pass # drawnDotCount won't be increased.
    def drawPointUsingComparison(mainPoint=None, comparisonPoint=None):
        drawPointUsingMask(mainPoint=mainPoint, mask=[True, mainPoint.real>comparisonPoint.real, mainPoint.imag>comparisonPoint.imag])
    
    
    
    visitPointListEcho = Echo(length=2)
    
    cellDescriptionGen = camera.seed_settings.iter_cell_descriptions(centered=False)
    
    print("done initializing.")
    
    for i, (x, y, seed) in enumerate(cellDescriptionGen):
        
        if (x==0):
            if banded and (y%(camera.seed_settings.grid_size[1]//IMAGE_BAND_COUNT) == 0):
                SET_LIVE_STATUS("drawing...")
                specializedDraw()
                SET_LIVE_STATUS("saving...")
                save_screenshot_as(name_prefix=output_name+"{}of{}rows{}of{}dotsdrawn_".format(y, camera.seed_settings.grid_size[1], drawingStats["drawnDotCount"], drawingStats["dotCount"]))
                SET_LIVE_STATUS("saved...")
            if CAPTION_RATE_LIMITER.get_judgement():
                SET_LIVE_STATUS("y={}".format(y))
                
        # old: journey = journeyFun(seed); constrainedJourney = [item for item in gen_constrain_journey(journey, iter_limit, escape_radius)] ...
        constrainedJourney = list(gen_suppress_exceptions(itertools.islice(escstopJourneyFun(seed), 0, iter_limit+1), esc_exceptions))
        
        if buddha_type == "jbb":
            seedIsInSet = True
        else:
            seedIsInSet = not (len(constrainedJourney) >= iter_limit)
            if buddha_type == "bb":
                pass
            elif buddha_type == "abb":
                seedIsInSet = not seedIsInSet
            else:
                assert False, "invalid buddha type {}.".format(buddha_type)
        
        if not seedIsInSet:
            visitPointListEcho.push([]) # don't let old visitPointList linger, it is no longer the one from the previous seed.
            continue
        else:
            
            # quarterbeveledJourney = gen_path_seg_quarterbevel_12x(constrainedJourney)
            # journeyWithTrackedDecayingMean = gen_track_decaying_mean(constrainedJourney, feedback=0.5)
            # journeyAndDecayingMeanSeqLadderRungSelfIntersections = gen_seg_seq_self_intersections(journeyWithTrackedDecayingMean, intersection_fun=SegmentGeometry.segment_intersection)
            # journeySelfNonIntersections = gen_path_self_non_intersections(constrainedJourney, intersection_fun=SegmentGeometry.segment_intersection)
            
            # journeySelfIntersections = gen_path_self_intersections(constrainedJourney, intersection_fun=SegmentGeometry.segment_intersection)
            # doubleJourneySelfIntersections = gen_path_self_intersections(journeySelfIntersections, intersection_fun=SegmentGeometry.segment_intersection)
            
            # modifiedJourney = gen_recordbreakers(journeySelfIntersections, score_fun=abs)
            # modifiedJourney = (point-seed for point in constrainedJourney[1:])
            # recordBreakersJourney = gen_multi_recordbreakers(constrainedJourney[1:], score_funs=[abs, inv_abs_of, (lambda inputVal: get_complex_angle(ensure_nonzero(inputVal)))])
            
            # shuffledJourney, constrainedJourney = (constrainedJourney, None); random.shuffle(shuffledJourney)
            
            #gspFromOriginJourney, constrainedJourney = (constrainedJourney, None); sort_to_greedy_shortest_path_order(gspFromOriginJourney); assert gspFromOriginJourney[0] == 0
            #gspFromOriginJourneySelfIntersections = gen_path_self_intersections(gspFromOriginJourney, intersection_fun=SegmentGeometry.segment_intersection)
            
            # sortedByAbsJourney, constrainedJourney = (constrainedJourney, None); list.sort(sortedByAbsJourney, key=abs)
            # sortedByAbsJourneySelfIntersections = gen_path_self_intersections(sortedByAbsJourney, intersection_fun=SegmentGeometry.segment_intersection)
            
            # sortedBySeedmanhdistJourney, constrainedJourney = (constrainedJourney[1:], None); list.sort(sortedBySeedmanhdistJourney, key=(lambda pt: SegmentGeometry.complex_manhattan_distance(seed, pt))); assert sortedBySeedmanhdistJourney[0]==seed
            # sortedBySeedmanhdistJourneySelfIntersections = gen_path_self_intersections(sortedBySeedmanhdistJourney, intersection_fun=SegmentGeometry.segment_intersection)
            
            """
            zjfiJourneyToFollow = sortedByAbsJourneySelfIntersections # skip first item here if necessary.
            zjfiJourneyToAnalyze = sortedByAbsJourneySelfIntersections
            # zjfiFoundationSegsToUse = [(complex(0,0), seed), (seed, zjfiJourneyToAnalyze[-1]), (complex(0,0), zjfiJourneyToAnalyze[-1])]  assert escape_radius==2.0
            zjfiFoundationSegsToUse = [(seed, max(escape_radius,2.0)*10.0*get_normalized(seed)), (complex(0,seed.imag), seed), (complex(seed.real,0), seed)]; assert camera.view.size.real < 10, "does a new spoke length for foundation tests need to be chosen?"
            zippedJourneyFoundationIntersections = gen_path_zipped_multi_seg_intersections(zjfiJourneyToFollow, reference_segs=zjfiFoundationSegsToUse, intersection_fun=SegmentGeometry.segment_intersection); # assert len(zjfiJourneyToAnalyze) < iter_limit, "bad settings! is this a buddhabrot, or is it incorrectly an anti-buddhabrot or a joint-buddhabrot?"; assert zjfiJourneyToAnalyze[0] == complex(0,0), "what? bad code?";
            """
            
            limitedVisitPointGen = itertools.islice(constrainedJourney, 0, point_limit)
            visitPointListEcho.push([item for item in limitedVisitPointGen])
        
        # non-differential mode:
        
        for ii, currentItem in enumerate(visitPointListEcho.current):
            # drawZippedPointsToChannels(currentItem)
            # drawPointUsingMask(mainPoint=currentItem[0], mask=currentItem[1])
            drawPointUsingComparison(mainPoint=currentItem, comparisonPoint=seed)
        
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
    print("doing final draw...")
    specializedDraw()
    print("doing final screenshot...")
    save_screenshot_as(name_prefix=output_name)
    print("do_buddhabrot done.")
    SET_LIVE_STATUS("done.")













def quadrilateral_is_convex(points):
    assert len(points) == 4
    return segments_intersect((points[0], points[2]), (points[1], points[3]))


def enumerate_to_depth(data, depth=None):
    assert depth > 0
    if depth == 1:
        for pair in enumerate(data): # return can't be used because yield appears in other branch. This does NOT produce error messages in python 3.8.10.
            yield pair
    else:
        assert depth > 1
        for i, item in enumerate(data):
            for longItem in enumerate_to_depth(item, depth=depth-1):
                yield (i,) + longItem
assert_equal(list(enumerate_to_depth([5,6,7,8], depth=1)), [(0,5), (1,6), (2,7), (3,8)])
assert_equal(list(enumerate_to_depth([[5,6],[7,8]], depth=2)), [(0,0,5), (0,1,6), (1,0,7), (1,1,8)])


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

def mean(input_seq):
    sumSoFar = 0
    i = -1
    for i, item in enumerate(input_seq):
        sumSoFar += item
    itemCount = i + 1
    if itemCount == 0:
        return 0
    assert itemCount > 0
    return sumSoFar / float(itemCount)


"""
PanelCell = collections.NamedTuple("PanelCellName", ["seed", "current_z", "previous_z", "set_membership"])
testPanelCell = PanelCell(0+0j,1+1j,2+2j,True)
assert testPanelCell.seed == 0+0j
del testPanelCell
"""
i_SEED, i_PREVIOUS_Z, i_CURRENT_Z, i_ISINSET = (0, 1, 2, 3)


@measure_time_nicknamed("create_panel")
def create_panel(seed_settings, iter_limit=None, escape_radius=None, buddhabrot_set_type=None, centered_sample=None):
    print("constructing empty panel...")

    statusRateLimiter = PygameDashboard.RateLimiter(3.0)

    assert seed_settings.grid_size[0] <= 4096, "make sure there is enough memory for this!"
    panel = construct_data(seed_settings.grid_size[::-1], default_value=None)
    arglessItercountFun = (lambda: c_to_mandel_itercount_fast(seed, iter_limit, escape_radius))
    
    print("populating panel...")
    
    assert abs(seed_settings.graveyard_point) > escape_radius
    for x, y, seed in seed_settings.iter_cell_descriptions(centered=centered_sample):
        panelCell = [seed, 0.0+0.0J, 0.0+0.0J, None]
        panelCell[i_ISINSET] = check_bb_containedness(argless_itercount_fun=arglessItercountFun, iter_limit=iter_limit, buddhabrot_set_type=buddhabrot_set_type)
        panel[y][x] = panelCell
        if x == 0 and statusRateLimiter.get_judgement():
            print("create_panel: {}%...".format(str(int(float(100*y)/seed_settings.grid_size[1])).rjust(2," ")))
            
    print("done creating panel.")
    return panel




@measure_time_nicknamed("do_panel_buddhabrot")
def do_panel_buddhabrot(camera, iter_limit=None, output_interval_iters=1, blank_on_output=True, count_scale=1, escape_radius=None, buddhabrot_set_type="bb"):
    assert iter_limit is not None
    assert buddhabrot_set_type in {"bb", "jbb", "abb"}
    
    # outputColorSummary = "R012outofsetneighG3outofsetneighB4outofsetneigh"
    # outputColorSummary = "top(RguestpaircmidptbothinsetGoneinsetBneitherinset)bottom(endpt)"
    outputGenerationSummary = "neighborPathSimultaneousCross"
    output_name="panel_{}_{}_{}pos{}fov{}esc{}itr{}biSuper{}count_{}_{}_".format(buddhabrot_set_type, outputGenerationSummary, camera.view.center_pos, camera.view.size, escape_radius, iter_limit, camera.bidirectional_supersampling, count_scale, COLOR_SETTINGS_SUMMARY_STR, ("blankOnOut" if blank_on_output else "noBlankOnOut"))
    
    print("creating visitCountMatrix...")
    visitCountMatrix = construct_data(camera.screen_settings.grid_size[::-1], default_value=[0,0,0])
    assert tuple(shape_of(visitCountMatrix)) == camera.screen_settings.grid_size[::-1]+(3,)
    
    
    def pointToVisitCountMatrixCell(point): # DUPLICATE CODE  DO NOT MODIFY
        return camera.screen_settings.complex_to_item(visitCountMatrix, point, centered=False)
    def drawPointUsingMask(mainPoint=None, mask=None): # DUPLICATE CODE  DO NOT MODIFY
        # drawingStats["dotCount"] += 1
        try:
            vec_add_scalar_masked(pointToVisitCountMatrixCell(mainPoint), count_scale, mask)
            # drawingStats["drawnDotCount"] += 1
        except IndexError:
            pass # drawnDotCount won't be increased.
    def drawPointUsingComparison(mainPoint=None, comparisonPoint=None): # DUPLICATE CODE  DO NOT MODIFY
        drawPointUsingMask(mainPoint=mainPoint, mask=[True, mainPoint.real>comparisonPoint.real, mainPoint.imag>comparisonPoint.imag])
    
    
    def specializedDraw():
        draw_squished_ints_to_screen(visitCountMatrix, access_order="yxc")
    def specializedDrawAndScreenshot(name_prefix=None):
        assert name_prefix is not None
        specializedDraw()
        save_screenshot_as(name_prefix=name_prefix)
    def blankIfNeeded():
        if blank_on_output:
            fill_data(visitCountMatrix, 0)
    
    def iteratePanelPoints():
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            if panelCell[i_CURRENT_Z] is not None:
                panelCell[i_PREVIOUS_Z], panelCell[i_CURRENT_Z] = (panelCell[i_CURRENT_Z], panelCell[i_CURRENT_Z]**2 + panelCell[i_SEED])
                if abs(panelCell[i_CURRENT_Z]) > escape_radius:
                    panelCell[i_PREVIOUS_Z], panelCell[i_CURRENT_Z] = (None, None) # (camera.seed_settings.graveyard_point, camera.seed_settings.graveyard_point)
    
    panel = create_panel(camera.seed_settings, iter_limit=iter_limit, escape_radius=escape_radius, buddhabrot_set_type=buddhabrot_set_type, centered_sample=False)
    
    """
    hotelGrid = construct_data(camera.seed_settings.grid_size[::-1], default_value=[])
    assert hotelGrid[0][0] is not hotelGrid[0][1]
    print("done creating hotelGrid.")
    """
    
    for iter_index in range(0, iter_limit):
    
        if (iter_index % output_interval_iters) == 0:
            specializedDrawAndScreenshot(name_prefix=output_name+"{}of{}itrs_".format(iter_index, iter_limit))
            blankIfNeeded()
            assert blank_on_output, "not ready yet?"
    
        iteratePanelPoints()
                
        # panel_brot_draw_panel_based_on_neighbors_in_set(seed_settings=seed_settings, panel=panel, visit_count_matrix=visitCountMatrix, count_scale=count_scale, centered_sample=False)
        
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            if y in (0, len(panel)-1) or x in (0,len(panel[0])-1):
                continue
            if panelCell[i_CURRENT_Z] is None:
                continue
            neighboringPanelCells = [panel[neighborY][neighborX] for neighborX,neighborY in ((x, y-1), (x+1,y), (x,y+1), (-x, y))]
            for neighboringPanelCell in neighboringPanelCells:
                if neighboringPanelCell[i_CURRENT_Z] is None:
                    continue
                intersectionPoint = SegmentGeometry.segment_intersection((panelCell[i_PREVIOUS_Z], panelCell[i_CURRENT_Z]), (neighboringPanelCell[i_PREVIOUS_Z], neighboringPanelCell[i_CURRENT_Z]))
                if intersectionPoint is not None:
                    drawPointUsingComparison(mainPoint=intersectionPoint, comparisonPoint=panelCell[i_SEED])
        
        # hotel code:
        """
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            try:
                hotel = camera.seed_settings.complex_to_item(hotelGrid, panelCell[i_CURRENT_Z], centered=False)
            except IndexError:
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
                    except IndexError:
                        continue # the point is not on the screen.
                    vec_add_scalar_masked(visitCountMatrixCell, count_scale, mask)
            
        for hotelY, hotelX, hotel in enumerate_to_depth(hotelGrid, depth=2):
            hotel.clear()
        """
                
    specializedDrawAndScreenshot(name_prefix=output_name)
    blankIfNeeded()
                
                
                
                
def panel_brot_draw_close_encounters():
    raise NotImplementedError("?")


def panel_brot_draw_panel_based_on_neighbors_in_set(seed_settings=None, panel=None, visit_count_matrix=None, count_scale=None, centered_sample=None):
    for y, panelRow in itertools.islice(enumerate(panel), 1, len(panel)-1):
        for x, panelCell in itertools.islice(enumerate(panelRow), 1, len(panelRow)-1):
            if not panelCell[i_ISINSET]: # if this point isn't in the desired set:
                continue

            try:
                visitCountMatrixCell = seed_settings.complex_to_item(visit_count_matrix, panelCell[i_CURRENT_Z], centered_sample=centered_sample)
            except IndexError:
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





def SET_LIVE_STATUS(status_text):
    try:
        pygame.display.set_caption("z. " + status_text)
    except Exception as e:
        print("couldn't set caption to: {}. error: {}.".format(repr(status_text), e))

def save_screenshot_as(*args, **kwargs):
    pygame.display.flip()
    save_surface_as(screen, *args, **kwargs)
    
def draw_squished_ints_to_screen(*args, **kwargs):
    draw_squished_ints_to_surface(screen, *args, **kwargs)
    pygame.display.flip()




pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((4096, 4096))
IMAGE_BAND_COUNT = (
    4 if screen.get_size()[1] <= 128 else (
    16 if screen.get_size()[1] <= 512 else
    32))

assert screen.get_size()[0] == screen.get_size()[1], "are you sure about that?"
assert screen.get_size()[0] in {4,8,16,32,64,128,256,512,1024,2048,4096}, "are you sure about that?"

COLOR_SETTINGS_SUMMARY_STR = "color(atan)"
OUTPUT_FOLDER = "outbox2/"




def main():

    #test_abberation([0], 0, 16384)
    # for biSup, iterLim, ptLim in [(1024)]
    # "((abs(c*0.25)+0.015625)**-1)"
    # (1.75+0.5*sin(iter_index))
    # complex({}*sin(z.imag+c.real), tan(z.real+c.imag))+c
    # for steppedVal in [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0]:
    # for steppedVal in ComplexGeometry.float_range(1, 8, 0.03125):
    do_buddhabrot(Camera(View(0+0j, 16+16j), screen_size=screen.get_size(), bidirectional_supersampling=4), iter_limit=256, point_limit=256, count_scale=2, init_formula="w,z=c,c", iter_formula="w,z=w**z-c,z**w+c", esc_test="(abs(w)>16)or(abs(z)>16)", esc_exceptions=(OverflowError,ZeroDivisionError), buddha_type="bb", banded=True)
    # do_panel_buddhabrot(Camera(View(0+0j, 4+4j), screen_size=screen.get_size(), bidirectional_supersampling=2), iter_limit=1024, output_interval_iters=1, count_scale=1, escape_radius=256.0)

    PygameDashboard.stall_pygame(preferred_exec=THIS_MODULE_EXEC)








                
print("done testing.")

if __name__ == "__main__":
    main()
    print("""
                   # #
     ##            # #        ##  #
    #    ### ###  ## ##      #    #
    # ## # # # # # # # # # # ###  #
    #  # ### ###  ## ##   #  #     
     ##                 ##    ##  #
     
     
    """)


# ready to go