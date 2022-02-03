#!/usr/bin/python3

import time
import math
import itertools
import collections
import copy

import pygame

from ColorTools import atan_squish_unsigned, automatic_color, squish_color
import SegmentGeometry



pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((1024, 1024))


assert screen.get_size()[0] == screen.get_size()[1], "are you sure about that?"
assert screen.get_size()[0] in {4,8,16,32,64,128,256,512,1024,2048,4096}, "are you sure about that?"


import PygameDashboard

def this_module_exec(string):
    exec(string)
# PygameDashboard.parent_module_exec = this_module_exec



def assert_equal(thing0, thing1):
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)


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

    

    

def screenshot(name_prefix="", name=None):
    startTime = time.perf_counter()
    if name is None:
        name = "{}{}.png".format(time.perf_counter(), str(screen.get_size()).replace(", ","x"))
    usedName = name_prefix + name
    print("saving file {}.".format(usedName))
    pygame.image.save(screen, usedName)
    print("saving took {} seconds.".format(time.perf_counter()-startTime))
    
    
def enforce_tuple_length(input_tuple, length, default=None):
    assert type(input_tuple) == tuple
    if len(input_tuple) == length:
        return input_tuple
    elif len(input_tuple) < length:
        return input_tuple + tuple(default for i in range(length-len(input_tuple)))
    else:
        return input_tuple[:length]
    



def draw_squished_ints_to_screen(channels, access_order=None):
    # maybe this method shouldn't exist. Maybe image creation should happen in another process, like photo.py in GeodeFractals.
    startTime = time.time()
    
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
                if x==0 and y%128 == 0:
                    pygame.display.flip()
                color = enforce_tuple_length(tuple(int(atan_squish_unsigned(colorDataGetter(x, y, chi), 255)) for chi in range(cSize)), 3, default=0)
                # assert max(color) < 256
                # assert min(color) >= 0
                screen.set_at((x, y), color)
    except IndexError as ie:
        print("index error when (x, y)=({}, {}): {}.".format(x, y, ie))
        exit(1)
    pygame.display.flip()
    print("drawing squished ints to screen took {} seconds.".format(time.time()-startTime))
            


def range2d(width, height):
    for y in range(height):
        for x in range(width):
            yield x, y


"""
def capture_exits(input_fun):
    def modifiedFun(*args, **kwargs):
        try:
            input_fun(*args, **kwargs)
        except KeyboardInterrupt:
            print("KeyboardInterrupt intercepted.")
            return
    return modifiedFun"""
            
            
def gen_track_previous(input_seq):
    previousItem = None
    for item in input_seq:
        yield (previousItem, item)
        previousItem = item
        
        
def gen_track_previous_tuple_flatly(input_seq):
    previousTuple = None
    for item in input_seq:
        if not isinstance(item, tuple):
            item = (item,)
        if previousTuple is None:
            yield (None,)*len(item) + item
        else:
            yield previousTuple + item
        previousTuple = item
        assert isinstance(previousTuple, tuple)
        
        
def enumerate_flatly(input_seq, start=0):
    for i, item in enumerate(input_seq, start=start):
        if isinstance(item, tuple):
            yield (i,) + item
        else:
            yield (i, item)
            
            
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










def c_to_mandel_itercount_fast(c, iter_limit, escape_radius):
    z = 0.0
    for i in range(iter_limit):
        if abs(z) >= escape_radius:
            return i
        z = z**2.0 + c
    return None


def c_to_mandel_journey(c):
    z = 0+0j
    for i in itertools.count(0):
        yield z
        z = z**2 + c
        
        
def c_to_mandel_journey_abberated_by_addition(c, abberation_seq):
    z = 0
    yield z
    for i, abber in itertools.zip_longest(itertools.count(0), abberation_seq):
        z = z**2 + c
        if abber is not None:
            z += abber
        yield z
        
        
        
        
def journey_to_itercount(journey, iter_limit, escape_radius): #outdated.
    for i, point in enumerate(journey):
        if i >= iter_limit:
            return -1
        if abs(point) > escape_radius:
            return i
    assert False, "incomplete journey."
    
    
"""
def c_and_journey_fun_to_itercount(journey_fun, c, iter_limit, escape_radius): #outdated.
    journey = journey_fun(c)
    itercount = journey_to_itercount(journey, iter_limit, escape_radius)
    return itercount
        """
        

def constrain_journey(journey, iter_limit, escape_radius):
    for i, point in enumerate(journey):
        yield point
        if i >= iter_limit:
            return
        if abs(point) > escape_radius:
            return
    assert False, "incomplete journey."


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
    
    
    

"""    
def seg1_is_on_both_sides_of_seg0(seg0, seg1):
    seg0rise = (seg0[1].imag-seg0[0].imag)
    seg0run = (seg0[1].real-seg0[0].real)
    if seg0run == 0:
        seg0run = ZERO_DIVISION_NUDGE # avoid zero division errors later.
    seg0direction = (seg0[1]-seg0[0])
    # seg0slope = seg0rise/seg0run
    
    seg1startRealOffset = (seg1[0].real-seg0[0].real)
    seg1startAlignT = seg1startRealOffset/seg0run
    seg1startAlignPt = seg0[0] + seg1startAlignT*seg0direction
    seg1startSide = (seg1startAlignPt.imag < seg1[0].imag)
    
    seg1endRealOffset = (seg1[1].real-seg0[0].real)
    seg1endAlignT = seg1endRealOffset/seg0run
    seg1endAlignPt = seg0[0] + seg1endAlignT*seg0direction
    seg1endSide = (seg1endAlignPt.imag < seg1[1].imag)
    
    # seg1endAlignT = (seg1[1].real-seg0[0].real)/seg0run
    # seg1endAlignPt = seg0[0] + seg1endAlignT*seg0slope
    # seg1endSide = (seg1endAlignPt.imag < seg1[1].imag)
    
    return seg1startSide != seg1endSide
"""
    


"""
def count_intersections(constrained_journey): #could be made to use much less memory.
    result = 0
    knownSegs = []
    for previousPoint, point in gen_track_previous(constrained_journey):
        if previousPoint is not None:
            currentSeg = (previousPoint, point)
            knownSegs.append(currentSeg)
            for oldKnownSeg in knownSegs[:-1]: #don't include the last one just appended.
                if segments_intersect(currentSeg, oldKnownSeg):
                    result += 1
    return result"""
    
def gen_intersections(constrained_journey, intersection_fun=None): #could use less memory.
    knownSegs = []
    for previousPoint, point in gen_track_previous(constrained_journey):
        assert isinstance(point, complex)
        if previousPoint is not None:
            currentSeg = (previousPoint, point)
            knownSegs.append(currentSeg)
            for oldKnownSeg in knownSegs[:-1]: #don't include the last one just appended.
                intersection = intersection_fun(currentSeg, oldKnownSeg)
                if intersection is not None:
                    yield intersection

    
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
                
                

def get_seeds(screen_size: tuple, camera_pos: complex, view_size: complex, centered_sample=None):
    assert centered_sample is not None
    for y in range(screen_size[1]):
        for x in range(screen_size[0]):
            yield (x, y, screen_to_complex((x,y), screen_size, camera_pos, view_size, centered_sample=centered_sample))
            
def calc_pixel_size(screen_size, view_size):
    return (view_size.real/screen_size[0]) + (view_size.imag/screen_size[1])*1.0j
            
def calc_camera_properties(screen_size: tuple, camera_pos: complex, view_size: complex, centered_sample=None):
    assert centered_sample is not None, "this arg might not even be needed."
    assert centered_sample == False, "not implemented yet."
    camera_corner_pos = camera_pos - (view_size/2.0)
    pixel_size = calc_pixel_size(screen_size, view_size)
    return (camera_corner_pos, pixel_size)
    
def screen_to_complex(screen_coords: tuple, screen_size: tuple, camera_pos: complex, view_size: complex, centered_sample=None):
    assert centered_sample is not None
    camera_corner_pos, pixel_size = calc_camera_properties(screen_size, camera_pos, view_size, centered_sample=centered_sample)
    return camera_corner_pos + screen_coords[0]*pixel_size.real + screen_coords[1]*pixel_size.imag*1.0j + pixel_size*0.5*centered_sample
    
def complex_to_screen(complex_coord, screen_size, camera_pos, view_size, centered_sample=None):
    assert centered_sample is not None
    camera_corner_pos, pixel_size = calc_camera_properties(screen_size, camera_pos, view_size, centered_sample=centered_sample)
    posRelToCorner = complex_coord - camera_corner_pos
    y = posRelToCorner.imag/pixel_size.imag
    x = posRelToCorner.real/pixel_size.real
    assert centered_sample==False, "not implemented yet."
    return (int(x), int(y))
    
assert complex_to_screen(screen_to_complex((56,78), (123,456), 8+8j, 4+4j, centered_sample=False), (123,456), 8+8j, 4+4j, centered_sample=False) == (56, 78)



    
    
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
    
"""
def modify_visit_count_matrix(visit_count_matrix, input_seq, camera_pos, view_size, count_scale):
    for i, item in input_seq:
        complexPoint, redInc, greenInc, blueInc = item
        screenPixelX, screenPixelY = complex_to_screen(point, screen.get_size(), camera_pos, view_size)
        try:
            currentCell = visit_count_matrix[screenPixelY][screenPixelX]
            if redInc:
                currentCell[0] += count_scale
            if greenInc:
                currentCell[1] += count_scale
            if blueInc:
                currentCell[2] += count_scale
        except IndexError as ie:
            pass
"""

def scale_size(input_size, input_scale):
    assert len(input_size) == 2
    assert isinstance(input_scale, int)
    return (input_size[0]*input_scale, input_size[1]*input_scale)
    
    
class SeedSettings:
    # the slowdown from the use of this class is only about 7%. I think it is worth it.
    # the first seq to use this class was at time 1641086680.
    
    def __init__(self, camera_pos, view_size, screen_size, bidirectional_supersampling=1):
        assert screen_size == screen.get_size()
        assert all(is_round_binary(item) for item in screen_size)
        self.screen_size = tuple(item for item in screen_size) # make it a tuple as a standard for equality tests in other places.
        self.bidirectional_supersampling = assure_round_binary(bidirectional_supersampling)
        self.camera_pos = camera_pos
        assert view_size.real > 0.0
        assert view_size.imag > 0.0
        self.view_size = view_size
        
        self.supersize = scale_size(self.screen_size, self.bidirectional_supersampling)
        self.pixel_size = calc_pixel_size(self.screen_size, self.view_size)
        self.graveyard_point = (1+abs(self.bidirectional_supersampling))*(16.5 + abs(self.camera_pos) + abs(2.0*self.view_size)) # a complex coordinate that will never appear on camera. Make it so large that there is no doubt.
        
    def screen_to_complex(self, screen_coord, centered_sample=None):
        # assert centered_sample is not None
        return screen_to_complex(screen_coord, self.screen_size, self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def sample_coord_to_complex(self, sample_coord, centered_sample=None):
        # assert centered_sample is not None
        return screen_to_complex(sample_coord, self.supersize, self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def complex_to_anygrid_item(self, data, complex_coord, grid_size=None, centered_sample=None):
        x, y = self.complex_to_anygrid(complex_coord, grid_size=grid_size, centered_sample=centered_sample)
        return data[y][x]
        
    def complex_to_screen_item(self, data, complex_coord, centered_sample=None):
        return self.complex_to_anygrid_item(data, complex_coord, grid_size=self.screen_size, centered_sample=centered_sample)
        
    def complex_to_anygrid(self, complex_coord, grid_size=None, centered_sample=None):
        # centered_sample might not be logically needed for the answer to this question, depending on how the screen is defined in future versions of the program.
        return complex_to_screen(complex_coord, grid_size, self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def complex_to_screen(self, complex_coord, centered_sample=None):
        return self.complex_to_anygrid(complex_coord, self.screen_size, centered_sample=centered_sample)
        
    def complex_is_on_screen(self, complex_coord, centered_sample=None):
        assert centered_sample is not None
        if centered_sample != True:
            raise NotImplementedError("off-center samples?")
        viewQuadrantSize = self.view_size / 2.0
        realAligned = (complex_coord.real > self.camera_pos.real - viewQuadrantSize.real) and (complex_coord.real < self.camera_pos.real + viewQuadrantSize.real)
        imagAligned = (complex_coord.imag > self.camera_pos.imag - viewQuadrantSize.imag) and (complex_coord.imag < self.camera_pos.imag + viewQuadrantSize.imag)
        return realAligned and imagAligned
        
    def complex_to_sample_coord(self, complex_coord, centered_sample=None):
        assert False, "this method probably should never be used!"
        # return complex_to_screen(complex_coord, self.get_supersize(), self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def iter_sample_coords(self):
        return range2d(self.supersize[0], self.supersize[1])
        
    def iter_sample_descriptions(self, centered_sample=None):
        assert centered_sample is not None
        for x, y in self.iter_sample_coords():
            yield (x, y, self.sample_coord_to_complex((x,y), centered_sample=centered_sample))
        



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

"""
def compose_functions(function_list):
    assert len(function_list) > 0
    def inner(input_arg):
        result = None
        for fun in function_list:
            result = fun(result)
        return result
    return inner
    

def scaled_vec(vec0, scale):
    return [item*scale for item in vec0]
"""
    
    
def vec_add_vec_masked(vec0, vec1, mask):
    for i in range(len(vec0)):
        if mask[i]:
            vec0[i] += vec1[i]
        

def vec_add_scalar_masked(vec0, input_scalar, mask):
    for i in range(len(vec0)):
        if mask[i]:
            vec0[i] += input_scalar


def do_buddhabrot(seed_settings, iter_limit=None, point_limit=None, count_scale=1, escape_radius=4.0):
    assert iter_limit is not None
    assert point_limit is not None
    output_name="crosscrossbrot_below0.75pixSep_RallGincrvsleftBincivsleft_{}pos{}fov{}itrlim{}ptlim{}biSuper{}count_".format(seed_settings.camera_pos, seed_settings.view_size, iter_limit, point_limit, seed_settings.bidirectional_supersampling, count_scale)
    
    journeyFun = c_to_mandel_journey
    def specializedDraw():
        draw_squished_ints_to_screen(visitCountMatrix, access_order="yxc")
    
    visitCountMatrix = construct_data(screen.get_size()[::-1]+(3,), default_value=0)
    
    drawingStats = {"dotCount": 0, "drawnDotCount":0}
    
    pixelWidth = abs(seed_settings.screen_to_complex((0,0), centered_sample=False) - seed_settings.screen_to_complex((1,0), centered_sample=False))
    assert pixelWidth == seed_settings.pixel_size.real
    assert type(pixelWidth) == float
    assert pixelWidth < 0.1, "is the screen really that small?"
    
    visitPointListEcho = Echo(length=2)
    
    for i, x, y, seed in enumerate_flatly(seed_settings.iter_sample_descriptions(centered_sample=False)):
        
        if x==0 and y%128 == 0:
            specializedDraw()
            if y%128 == 0 or y == seed_settings.screen_size[1]//2:
                screenshot(name_prefix=output_name+"{}of{}rows{}of{}dotsdrawn_".format(y, seed_settings.supersize[1], drawingStats["drawnDotCount"], drawingStats["dotCount"]))
                
        journey = journeyFun(seed)
        constrainedJourney = [item for item in constrain_journey(journey, iter_limit, escape_radius)]
        
        if len(constrainedJourney) >= iter_limit:
            visitPointListEcho.push([]) # don't let old visitPointList linger, it is no longer the one from the previous seed.
            continue
        else:
            journeySelfIntersections = gen_intersections(constrainedJourney, intersection_fun=SegmentGeometry.segment_intersection)
            doubleJourneySelfIntersections = gen_intersections(journeySelfIntersections, intersection_fun=SegmentGeometry.segment_intersection)
            limitedVisitPointGen = itertools.islice(doubleJourneySelfIntersections, 0, point_limit)
            visitPointListEcho.push([item for item in limitedVisitPointGen])
        
        if i == 0:
            print("in differential mode, the first point's journey is not drawn.")
        else:
            comparisonMaskedVisitPointPairList = [pointPair for pointPair in zip(visitPointListEcho.previous, visitPointListEcho.current) if abs(pointPair[1]-pointPair[0]) < 0.75*pixelWidth]
            
            for ii, (leftPoint, centerPoint) in enumerate(comparisonMaskedVisitPointPairList):
                assert ii <= point_limit
                drawingStats["dotCount"] += 1
                try:
                    currentCell = seed_settings.complex_to_screen_item(visitCountMatrix, centerPoint, centered_sample=False)
                    vec_add_scalar_masked(currentCell, count_scale, [True, centerPoint.real>leftPoint.real, centerPoint.imag>leftPoint.imag])
                    drawingStats["drawnDotCount"] += 1
                except IndexError:
                    # drawnDotCount won't be increased.
                    pass
            # modify_visit_count_matrix(visitCountMatrix, ((curTrackPt, True, (curTrackPt.real>prevTrackPt.real), (curTrackPt.imag>prevTrackPt.imag)...
                
    specializedDraw()
    screenshot(name_prefix=output_name)



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




i_SEED, i_CURRENT_Z, i_PREVIOUS_Z, i_ISINSET = (0, 1, 2, 3)


def create_panel(seed_settings, iter_limit=None, escape_radius=None, buddhabrot_set_type=None, centered_sample=None):
    print("constructing empty panel...")

    statusRateLimiter = PygameDashboard.RateLimiter(3.0)

    assert seed_settings.supersize[0] <= 4096, "make sure there is enough memory for this!"
    panel = construct_data(seed_settings.supersize[::-1], default_value=None)
    
    print("populating panel...")
    
    assert abs(seed_settings.graveyard_point) > escape_radius
    for x, y, seed in seed_settings.iter_sample_descriptions(centered_sample=centered_sample):
        panelCell = [seed, 0.0+0.0J, 0.0+0.0J, None]
        panelCell[i_ISINSET] = check_bb_containedness(argless_itercount_fun=(lambda: c_to_mandel_itercount_fast(seed, iter_limit, escape_radius)),
            iter_limit=iter_limit, buddhabrot_set_type=buddhabrot_set_type,
        )
        panel[y][x] = panelCell
        if x == 0 and statusRateLimiter.get_judgement():
            print("create_panel: {}%...".format(str(int(float(100*y)/seed_settings.supersize[1])).rjust(2," ")))
        # assert seed_settings.complex_to_screen(seed, centered_sample=False) == (x, y) # not to screen... not if supersampling is allowed.
    # assert tuple(shape_of(panel)) == screen.get_size()[::-1]+(2,) # not true anymore.
    print("done creating panel.")
    return panel





def do_panel_buddhabrot(seed_settings, iter_limit=None, output_interval_iters=1, blank_on_output=True, count_scale=1, escape_radius=4.0, buddhabrot_set_type="bb"):
    assert iter_limit is not None
    assert buddhabrot_set_type in {"bb", "jbb", "abb"}
    
    # outputColorSummary = "R012outofsetneighG3outofsetneighB4outofsetneigh"
    outputColorSummary = "top(RguestpaircmidptbothinsetGoneinsetBneitherinset)bottom(endpt)"
    output_name="normal_{}_{}_{}pos{}fov{}itr{}biSuper{}count_{}_".format(buddhabrot_set_type, outputColorSummary, seed_settings.camera_pos, seed_settings.view_size, iter_limit, seed_settings.bidirectional_supersampling, count_scale, ("blankOnOut" if blank_on_output else "noBlankOnOut"))
    
    print("creating visitCountMatrix...")
    visitCountMatrix = construct_data(seed_settings.screen_size[::-1]+(3,), default_value=0)
    assert tuple(shape_of(visitCountMatrix)) == screen.get_size()[::-1]+(3,) # this isn't affected by supersampling! that comes later!
    
    
    def specializedDraw():
        draw_squished_ints_to_screen(visitCountMatrix, access_order="yxc")
    def specializedDrawAndScreenshot(name_prefix=None):
        assert name_prefix is not None
        specializedDraw()
        screenshot(name_prefix=name_prefix)
    
    
    panel = create_panel(seed_settings, iter_limit=iter_limit, escape_radius=escape_radius, buddhabrot_set_type=buddhabrot_set_type, centered_sample=False)
    
    
    hotelGrid = construct_data(seed_settings.supersize[::-1], default_value=[])
    assert hotelGrid[0][0] is not hotelGrid[0][1]
    print("done creating hotelGrid.")
    
    
    for iter_index in range(0, iter_limit):
    
        if (iter_index % output_interval_iters) == 0:
            specializedDrawAndScreenshot(name_prefix=output_name+"{}of{}itrs_".format(iter_index, iter_limit))
            if blank_on_output:
                fill_data(visitCountMatrix, 0)
            assert blank_on_output, "not ready yet?"
            # assert_equals(str(visitCountMatrix).replace("0","").replace("[","").replace("]","").replace(",","").replace(" ",""), "")
    
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            if abs(panelCell[i_CURRENT_Z]) > escape_radius:
                # using continue here without overwriting the value causes visual bugs. I don't know the reason yet.
                panelCell[i_CURRENT_Z] = seed_settings.graveyard_point
                # assert abs(panelCell[i_CURRENT]) > escape_radius
            else:
                panelCell[i_CURRENT_Z] = panelCell[i_CURRENT_Z]**2 + panelCell[i_SEED]
                
        # panel_brot_draw_panel_based_on_neighbors_in_set(seed_settings=seed_settings, panel=panel, visit_count_matrix=visitCountMatrix, count_scale=count_scale, centered_sample=False)
        
        for y, x, panelCell in enumerate_to_depth(panel, depth=2):
            try:
                hotel = seed_settings.complex_to_anygrid_item(hotelGrid, panelCell[i_CURRENT_Z], grid_size=seed_settings.supersize, centered_sample=False)
            except IndexError:
                continue # out of bounds
            hotel.append(panelCell[i_SEED])
        for hotelY, hotelX, hotel in enumerate_to_depth(hotelGrid, depth=2):
            if len(hotel) < 2:
                continue
                
            for guestA, guestB in gen_ordered_item_pairs(hotel):
                guestPairMidpoint = (guestA + guestB) / 2.0
                
                guestApanelCell, guestBpanelCell = [seed_settings.complex_to_anygrid_item(panel, guest, grid_size=seed_settings.supersize, centered_sample=False) for guest in (guestA, guestB)]
                guestAisInSet, guestBisInSet = (guestApanelCell[i_ISINSET], guestBpanelCell[i_ISINSET])
                mask = [(guestAisInSet == guestBisInSet == True), (guestAisInSet != guestBisInSet), (guestAisInSet == guestBisInSet == False)]
                
                for itemIsAllowed, item in [(guestA.imag > 0, guestA), (guestB.imag > 0, guestB), (guestPairMidpoint.imag < 0, guestPairMidpoint)]:
                    if not itemIsAllowed:
                        continue
                    try:
                        visitCountMatrixCell = seed_settings.complex_to_screen_item(visitCountMatrix, item, centered_sample=False)
                    except IndexError:
                        continue # the point is not on the screen.
                    vec_add_scalar_masked(visitCountMatrixCell, count_scale, mask)
            
        for hotelY, hotelX, hotel in enumerate_to_depth(hotelGrid, depth=2):
            hotel.clear()
            
                
    specializedDrawAndScreenshot(name_prefix=output_name)
    
    if blank_on_output:
        fill_data(visitCountMatrix, 0)
                
                
                
                
def panel_brot_draw_close_encounters():
    pass


def panel_brot_draw_panel_based_on_neighbors_in_set(seed_settings=None, panel=None, visit_count_matrix=None, count_scale=None, centered_sample=None):
    for y, panelRow in itertools.islice(enumerate(panel), 1, len(panel)-1):
        for x, panelCell in itertools.islice(enumerate(panelRow), 1, len(panelRow)-1):
            if not panelCell[i_ISINSET]: # if this point isn't in the desired set:
                continue

            try:
                visitCountMatrixCell = seed_settings.complex_to_screen_item(visit_count_matrix, panelCell[i_CURRENT_Z], centered_sample=centered_sample)
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






                
print("done testing.")

#test_abberation([0], 0, 16384)
do_buddhabrot(SeedSettings(0+0j, 4+4j, screen.get_size(), bidirectional_supersampling=4), iter_limit=256, point_limit=256, count_scale=4)
# do_panel_buddhabrot(SeedSettings(0+0j, 4+4j, screen.get_size(), bidirectional_supersampling=1), iter_limit=1024, output_interval_iters=1, count_scale=8) # <------

#test_nonatree_mandelbrot(-0.5+0j, 4+4j, 64, 6)

PygameDashboard.stall_pygame(preferred_exec=this_module_exec)




"""
def solve_to_seq(start_args, rule_fun, termination_fun, iter_limit):
    state = 
    for i in range(iter_limit):
"""
"""
try:
    screenshot(name_prefix="screen_backup_")
    #if pygame.display.get_active():
    #    pygame.display.quit()
except Exception as e:
    raise e
    """