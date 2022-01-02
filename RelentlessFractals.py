#!/usr/bin/python3

import time
import math
import itertools

import pygame

from ColorTools import atan_squish_unsigned, automatic_color, squish_color

ZERO_DIVISION_NUDGE = 2**-64
MODULUS_OVERLAP_NUDGE = 2**-48

pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((1024, 1024))


assert screen.get_size()[0] == screen.get_size()[1], "are you sure about that?"
assert screen.get_size()[0] in {4,8,16,32,64,128,256,512,1024,2048,4096}, "are you sure about that?"


import PygameDashboard

def this_module_exec(string):
    exec(string)
PygameDashboard.parent_module_exec = this_module_exec




def shape_of(data_to_test):
    result = []
    while hasattr(data_to_test, "__len__"):
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

    
def assert_equals(thing0, thing1):
    if isinstance(thing0, complex) and isinstance(thing1, complex):
        if thing0-thing1 == 0:
            return
        elif abs(thing0-thing1) < 2**-36:
            # print("warning: {} and {} are supposed to be equal.".format(thing0, thing1))
            return
        print("this test is failing soon.")
    if isinstance(thing0, tuple) and isinstance(thing1, tuple):
        for i in range(max(len(thing0), len(thing1))):
            assert_equals(thing0[i], thing1[i])
        return
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)
    

def screenshot(name_prefix="", name=None):
    startTime = time.time()
    if name is None:
        name = "{}{}.png".format(time.time(), str(screen.get_size()).replace(", ","x"))
    usedName = name_prefix + name
    print("saving file {}.".format(usedName))
    pygame.image.save(screen, usedName)
    print("saving took {} seconds.".format(time.time()-startTime))
    
    
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
        return [default_value for i in range(size[0])]
    else:
        return [construct_data(size[1:], default_value=default_value) for i in range(size[0])]

assert_equals(shape_of(construct_data([5,6,7])), [5,6,7])


def c_to_mandel_itercount_fast(c, iter_limit, escape_radius):
    z = 0
    for i in range(iter_limit):
        if abs(z) >= escape_radius:
            return i
        z = z**2 + c
    return -1


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
    
    
    
def seg0_might_intersect_seg1(seg0, seg1):
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
    
    
print("extra assertions are turned off for segments_intersect because they are failing now.")

def segments_intersect(seg0, seg1, extra_assertions=False):
    result = (seg0_might_intersect_seg1(seg0, seg1) and seg0_might_intersect_seg1(seg1, seg0))
    if extra_assertions:
        if result:
            assert segment_intersection(seg0, seg1, extra_assertions=False) is not None
    return result

# tests for segments_intersect come later.



def cross_multiply(vec0, vec1):
    return vec0[0]*vec1[1] - vec0[1]*vec1[0]


def segment_intersection(seg0, seg1, extra_assertions=True):
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
    if abs(seg0intersection - seg1intersection) > 2.0**-64:
        return None
    if t < 0 or t > 1 or u < 0 or u > 1:
        return None
    if extra_assertions:
        assert segments_intersect(seg0, seg1, extra_assertions=False)
    return seg0intersection
    
assert not segments_intersect((1+1j, 2+1j), (1+2j, 2+2j))
assert segments_intersect((1+1j, 2+2j), (1+2j, 2+1j))
assert not segments_intersect((1+1j, 5+5j), (2+1j, 6+5j))
    
assert segment_intersection((1.0+1.0j, 1.0+3.0j), (0.0+2.0j, 2.0+2.0j)) == (1.0+2.0j)
assert segment_intersection((1.0+1.0j, 1.0+3.0j), (0.0+0.0j, 0.0+2.0j)) == None
assert segment_intersection((0+0j, 1+1j), (1+0j, 0+1j)) == (0.5+0.5j)


def get_complex_angle(c):
    if c.real == 0:
        c = c + ZERO_DIVISION_NUDGE
    return math.atan(c.imag/c.real) + (math.pi if c.real < 0 else (2*math.pi if c.imag < 0 else 0))

assert get_complex_angle(2+2j) == math.pi/4.0
assert get_complex_angle(-2+2j) == 3*math.pi/4.0
assert get_complex_angle(-2-2j) == 5*math.pi/4.0
assert get_complex_angle(2-2j) == 7*math.pi/4.0

assert get_complex_angle(1j) == math.pi/2.0
assert get_complex_angle(-1j) == 1.5*math.pi


def seg_is_valid(seg):
    return isinstance(seg,tuple) and all(isinstance(item, complex) for item in seg)


def polar_seg_is_valid(seg):
    assert seg_is_valid(seg)
    return (min(item.imag for item in seg) >= 0 and max(item.imag for item in seg) < 2*math.pi and item.real >= 0)
    
    
def assert_polar_seg_is_valid(seg):
    assert seg_is_valid(seg)
    for i, item in enumerate(seg):
        assert item.imag >= 0, (i, seg)
        assert item.imag < 2*math.pi, (i, seg)
        assert item.real >= 0, (i, seg)
        
        
def point_polar_to_rect(polar_pt):
    rectPt = polar_pt.real*(math.e**(polar_pt.imag*1j))
    return rectPt
    
def point_rect_to_polar(rect_pt):
    assert isinstance(rect_pt, complex)
    return (abs(rect_pt)+get_complex_angle(rect_pt)*1j)
    
def seg_rect_to_polar(seg):
    assert len(seg) == 2
    return tuple(point_rect_to_polar(seg[i]) for i in (0,1))
    
def seg_polar_to_rect(seg):
    assert len(seg) == 2
    return tuple(point_polar_to_rect(seg[i]) for i in (0,1))
    
#tests:
for testPt in [1+1j, -1+1j, -1-1j, 1-1j]:
    # assert_equals(testPt, point_rect_to_polar(point_polar_to_rect(testPt))). negative length is not fair.
    assert_equals(testPt, point_polar_to_rect(point_rect_to_polar(testPt)))
    

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
    assert_equals(seg_polar_to_rect(polarSeg0), seg0)
    assert_equals(seg_polar_to_rect(polarSeg1), seg1)
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
    assert_equals(seg0wrapPt is None, polarSeg0wrapPt is None)
    assert_equals(seg1wrapPt is None, polarSeg1wrapPt is None)

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
        

try:
    assert polar_space_segment_intersection((10+1j, 10+20j), (1+10j, 20+10j)) is not None
    assert polar_space_segment_intersection((1+1j, 2+2j), (1+2j, 2+1j)) is not None

    assert polar_space_segment_intersection((-10+1j, -10+20j), (-1+10j, -20+10j)) is not None
    assert polar_space_segment_intersection((-1+1j, -2+2j), (-1+2j, -2+1j)) is not None


    assert polar_space_segment_intersection((-100+10j, 100+10j), (-5-100j, -5+100j)) is not None
    assert polar_space_segment_intersection((-100+10j, 100+10j), (5-100j, 5+100j)) is not None

    assert_equals(polar_space_segment_intersection((-0.1+0.1j, -1+1j), (0+1j, -1+0j)), ((2.0**0.5)/2.0)*(-1+1j))
    assert_equals(polar_space_segment_intersection((0.1+0.1j, 1+1j), (0+1j, 1+0j)), ((2.0**0.5)/2.0)*(1+1j))
    assert_equals(polar_space_segment_intersection((0+0j, 1+1j), (0+1j, 1+0j)), ((2.0**0.5)/2.0)*(1+1j))
except AssertionError as ae:
    #print(ae.message)
    print(ae)


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
    
def gen_intersections(constrained_journey, intersection_fun=segment_intersection): #could use less memory.
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

    
def count_float_local_minima(input_seq): #does not recognize any minimum with more than one identical value in a row.
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
        self.view_size = view_size
    
    def get_supersize(self):
        return scale_size(self.screen_size, self.bidirectional_supersampling)
        
    def get_pixel_size(self):
        return calc_pixel_size(self.screen_size, self.view_size)
        
    def screen_to_complex(self, screen_coord, centered_sample=None):
        # assert centered_sample is not None
        return screen_to_complex(screen_coord, self.screen_size, self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def sample_coord_to_complex(self, sample_coord, centered_sample=None):
        # assert centered_sample is not None
        return screen_to_complex(sample_coord, self.get_supersize(), self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def complex_to_screen(self, complex_coord, centered_sample=None):
        # centered_sample might not be logically needed for the answer to this question, depending on how the screen is defined in future versions of the program.
        return complex_to_screen(complex_coord, self.screen_size, self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def complex_to_sample_coord(self, complex_coord, centered_sample=None):
        assert False, "this method probably should never be used!"
        # return complex_to_screen(complex_coord, self.get_supersize(), self.camera_pos, self.view_size, centered_sample=centered_sample)
        
    def iter_sample_coords(self):
        return range2d(self.get_supersize()[0], self.get_supersize()[1])
        
    def iter_sample_descriptions(self, centered_sample=None):
        assert centered_sample is not None
        for x, y in self.iter_sample_coords():
            yield (x, y, self.sample_coord_to_complex((x,y), centered_sample=centered_sample))
        
        

def do_buddhabrot(seed_settings, iter_limit, count_scale=1, escape_radius=4.0):
    output_name="crosscrossbrot_below0.5pixSep_RallGincrvsleftBincivsleft_{}pos{}fov{}itr{}biSuper{}count_".format(seed_settings.camera_pos, seed_settings.view_size, iter_limit, seed_settings.bidirectional_supersampling, count_scale)
    
    journeyFun = c_to_mandel_journey
    def specializedDraw():
        draw_squished_ints_to_screen(visitCountMatrix, access_order="yxc")
    
    visitCountMatrix = construct_data(screen.get_size()[::-1]+(3,), default_value=0)
    
    drawingStats = {"dotCount": 0, "drawnDotCount":0}
    
    pixelWidth = abs(seed_settings.screen_to_complex((0,0), centered_sample=False) - seed_settings.screen_to_complex((1,0), centered_sample=False))
    assert pixelWidth == seed_settings.get_pixel_size().real
    assert type(pixelWidth) == float
    assert pixelWidth < 0.1, "is the screen really that small?"
    
    for i, x, y, seed in enumerate_flatly(seed_settings.iter_sample_descriptions(centered_sample=False)):
    
        testSeed = screen_to_complex((x,y), seed_settings.get_supersize(), seed_settings.camera_pos, seed_settings.view_size, centered_sample=False) # DO NOT MODIFY YET.
        assert seed == testSeed
        
        if x==0 and y%512 == 0:
            specializedDraw()
            if y%512 == 0:
                screenshot(name_prefix=output_name+"{}of{}rows{}of{}dotsdrawn_".format(y, seed_settings.get_supersize()[1], drawingStats["drawnDotCount"], drawingStats["dotCount"]))
                
        journey = journeyFun(seed)
        constrainedJourney = [item for item in constrain_journey(journey, iter_limit, escape_radius)]
        
        if len(constrainedJourney) >= iter_limit:
            continue
        
        journeySelfIntersections = gen_intersections(constrainedJourney)
        doubleJourneySelfIntersections = gen_intersections(journeySelfIntersections)
        
        visitPointList = [item for item in doubleJourneySelfIntersections]
        
        if i == 0:
            print("in differential mode, the first point's journey is not drawn.")
            # assert x == 0
            # firstEverSeed = seed
        else:
            # if i == 1:
            #    assert x == 1
            #    # secondEverSeed = seed
            #    # sampleWidth = abs(secondEverSeed-firstEverSeed)
            # assert visitPointList[0] == 0, "differential mode code is not designed for this."
            limitedVisitPointList = [pointPair[1] for pointPair in zip(prevVisitPointList, visitPointList) if abs(pointPair[1]-pointPair[0]) < 0.5*pixelWidth]
            
            for ii, point in enumerate(limitedVisitPointList):
                screenPixel = seed_settings.complex_to_screen(point, centered_sample=False)
                drawingStats["dotCount"] += 1
                try:
                    currentCell = visitCountMatrix[screenPixel[1]][screenPixel[0]]
                    if True:
                        currentCell[0] += count_scale
                    if point.real > prevVisitPointList[ii].real:
                        currentCell[1] += count_scale
                    if point.imag > prevVisitPointList[ii].imag:
                        currentCell[2] += count_scale
                    drawingStats["drawnDotCount"] += 1
                except IndexError:
                    # drawnDotCount won't be increased.
                    pass
            # modify_visit_count_matrix(visitCountMatrix, ((curTrackPt, True, (curTrackPt.real>prevTrackPt.real), (curTrackPt.imag>prevTrackPt.imag)...
        prevVisitPointList = visitPointList
                
    specializedDraw()
    screenshot(name_prefix=output_name)



def quadrilateral_is_convex(points):
    assert len(points) == 4
    return segments_intersect((points[0], points[2]), (points[1], points[3]))



def do_panel_buddhabrot(seed_settings, iter_limit, output_interval_iters=1, count_scale=1, escape_radius=4.0):
    output_name="jointbuddhabrot_RallGconvexneighBconcaveneigh_{}pos{}fov{}itr{}biSuper{}count_".format(seed_settings.camera_pos, seed_settings.view_size, iter_limit, seed_settings.bidirectional_supersampling, count_scale)
    
    def specializedDraw():
        draw_squished_ints_to_screen(visitCountMatrix, access_order="yxc")
    
    assert seed_settings.get_supersize() == seed_settings.screen_size, "supersampling is currently not allowed in panel buddhabrot methods."
    
    visitCountMatrix = construct_data(seed_settings.screen_size[::-1]+(3,), default_value=0)
    assert tuple(shape_of(visitCountMatrix)) == screen.get_size()[::-1]+(3,)
    
    panel = construct_data(seed_settings.screen_size[::-1], default_value=None)
    assert tuple(shape_of(panel)) == screen.get_size()[::-1]
    for x, y, seed in seed_settings.iter_sample_descriptions(centered_sample=False):
        panel[y][x] = [seed, 0.0+0.0J]
        assert seed_settings.complex_to_screen(seed, centered_sample=False) == (x, y)
    assert tuple(shape_of(panel)) == screen.get_size()[::-1]+(2,)
        
    for iter_index in range(0, iter_limit):
    
        if (iter_index % output_interval_iters) == 0:
            specializedDraw()
            screenshot(name_prefix=output_name+"_{}of{}itrs_".format(iter_index, iter_limit))
    
        for y, panelRow in enumerate(panel):
            for x, panelCell in enumerate(panelRow):
                if abs(panelCell[1]) > escape_radius:
                    continue
                panelCell[1] = panelCell[1]**2 + panelCell[0]
                
        for y in range(1, len(panel)-1):
            panelRow = panel[y]
            for x in range(1, len(panelRow)-1):
                panelCell = panelRow[x]
                screenPixel = seed_settings.complex_to_screen(panelCell[1], centered_sample=False)
                thisPtIsConvex = quadrilateral_is_convex([panel[y-1][x][1], panel[y][x+1][1], panel[y+1][x][1], panel[y][x-1][1]])
                try:
                    visitCountMatrixCell = visitCountMatrix[screenPixel[1]][screenPixel[0]]

                    if True:
                        visitCountMatrixCell[0] += count_scale
                    if thisPtIsConvex:
                        visitCountMatrixCell[1] += count_scale
                    else:
                        visitCountMatrixCell[2] += count_scale
                except IndexError:
                    # the point is not on the screen.
                    pass
                
    specializedDraw()
    screenshot(name_prefix=output_name)
                
                

#test_abberation([0], 0, 16384)
# do_buddhabrot(SeedSettings(0+0j, 4+4j, screen.get_size(), bidirectional_supersampling=4), 32, count_scale=4)
do_panel_buddhabrot(SeedSettings(0+0j, 4+4j, screen.get_size(), bidirectional_supersampling=1), 256, output_interval_iters=16, count_scale=1)

#test_nonatree_mandelbrot(-0.5+0j, 4+4j, 64, 6)

PygameDashboard.stall_pygame()




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