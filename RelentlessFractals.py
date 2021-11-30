

import time
import math
import itertools

import pygame

from ColorTools import atan_squish_unsigned, automatic_color, squish_color

ZERO_DIVISION_NUDGE = 2**-64
MODULUS_OVERLAP_NUDGE = 2**-48

pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((128,128))

KEYBOARD_DIGITS = "0123456789"
KEYBOARD_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
KEYBOARD_SYMBOLS = " `~!@#$%^&*()-_=+[{]}\\|\\;:'\",<.>/?"
KEYBOARD_CHARS = KEYBOARD_DIGITS + KEYBOARD_LETTERS + KEYBOARD_SYMBOLS
KEYBOARD_LOWERS = "`1234567890-=[]\;',./"
KEYBOARD_UPPERS = "~!@#$%^&*()_+{}|:\"<>?"

def stall_pygame():
    print("stall.")
    ENTER = 13
    BACKSPACE = 8
    CAPSLOCK = 1073741881
    SHIFT = 1073742053
    capsLockIsOn = False
    shiftIsOn = False
    commandString = ""
    running = True
    while running:
        time.sleep(0.1)
        pygame.display.flip()
        for event in pygame.event.get():
            #print((event, dir(event), event.type))
            if event.type in [pygame.K_ESCAPE, pygame.KSCAN_ESCAPE, pygame.QUIT]:
                print("closing pygame as requested.")
                pygame.display.quit()
                running = False
                break
                
            if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
                if event.key == SHIFT:
                    if event.type == pygame.KEYDOWN:
                        shiftIsOn = True
                    if event.type == pygame.KEYUP:
                        shiftIsOn = False
                
            if event.type in [pygame.KEYDOWN]:
                if event.key == ENTER:
                    exec(commandString)
                    commandString = ""
                elif event.key == BACKSPACE:
                    commandString = commandString[:-1]
                elif event.key == CAPSLOCK:
                    capsLockIsOn = not capsLockIsOn
                else:
                    try:
                        newChar = chr(event.key)
                        assert newChar in KEYBOARD_CHARS
                        if shiftIsOn:
                            if newChar in KEYBOARD_LOWERS:
                                newChar = KEYBOARD_UPPERS[KEYBOARD_LOWERS.index(newChar)]
                            elif newChar in KEYBOARD_LETTERS:
                                newChar = newChar.upper()
                        commandString = commandString + newChar
                    except:
                        # print("problem with char with code {}.".format(key))
                        pass
                pygame.display.set_caption(commandString)
    print("end of stall.")
    
def assert_equals(thing0, thing1):
    if isinstance(thing0, complex) and isinstance(thing1, complex):
        if thing0-thing1 == 0:
            return
        elif abs(thing0-thing1) < 2**-36:
            #print("warning: {} and {} are supposed to be equal.".format(thing0, thing1))
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

def draw_squished_ints_to_screen(channels):
    startTime = time.time()
    for y in range(len(channels[0])):
        for x in range(len(channels[0][y])):
            if x==0 and y%128 == 0:
                pygame.display.flip()
            color = (tuple(int(atan_squish_unsigned(channels[chi][y][x], 255)) for chi in range(len(channels))) + (0, 0, 0))[:3]
            #assert max(color) < 256
            #assert min(color) >= 0
            screen.set_at((x,y), color)
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
        return [default_value for i in range(size[-1])]
    else:
        return [construct_data(size[:-1], default_value=default_value) for i in range(size[-1])]




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
    #print("returning {}.".format(result))
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
    
def segments_intersect(seg0, seg1):
    return (seg0_might_intersect_seg1(seg0, seg1) and seg0_might_intersect_seg1(seg1, seg0))

assert not segments_intersect((1+1j, 2+1j), (1+2j, 2+2j))
assert segments_intersect((1+1j, 2+2j), (1+2j, 2+1j))
assert not segments_intersect((1+1j, 5+5j), (2+1j, 6+5j))

def cross_multiply(vec0, vec1):
    return vec0[0]*vec1[1] - vec0[1]*vec1[0]

def segment_intersection(seg0, seg1):
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
    if abs(seg0intersection - seg1intersection) > 2.0**-16:
        return None
    if t < 0 or t > 1 or u < 0 or u > 1:
        return None
    return seg0intersection
    
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
            
def calc_camera_properties(screen_size: tuple, camera_pos: complex, view_size: complex):
    camera_corner_pos = camera_pos - (view_size/2)
    pixel_size = (view_size.real/screen_size[0]) + (view_size.imag/screen_size[1])*1.0j
    return (camera_corner_pos, pixel_size)
    
def screen_to_complex(screen_coords: tuple, screen_size: tuple, camera_pos: complex, view_size: complex, centered_sample=None):
    camera_corner_pos, pixel_size = calc_camera_properties(screen_size, camera_pos, view_size)
    return camera_corner_pos + screen_coords[0]*pixel_size.real + screen_coords[1]*pixel_size.imag*1.0j + pixel_size*0.5*centered_sample
    
def complex_to_screen(complex_coord, screen_size, camera_pos, view_size, centered_sample=False):
    camera_corner_pos, pixel_size = calc_camera_properties(screen_size, camera_pos, view_size)
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
        
class Nonacell:
    def __init__(self, data=None, view_pos=None, view_size=None):
        if data is None:
            data = construct_data((3,3), default_value=None)
            data = [[120, 130, 140], [150, 160, 170], [180, 190, 200]]
        assert len(data) == 3
        self.data = data
        self.parent = None
        self.view_pos, self.view_size = None, None
    def __getitem__(self, index):
        result = self.data[index]
        return result
    def get_child_pos(self, subX, subY):
        return self.view_pos + self.get_child_size().real*(subX-1) + self.get_child_size().imag*(subY-1)*1j
    def get_child_size(self):
        return self.view_size/3.0
        
def rasterize_nonatree(render_target, start_corner, end_corner, data_to_draw):
    #data_to_draw = [[20, 40, 60], [80, 100, 120], [140, 160, 180]]
    assert start_corner[0] < end_corner[0]
    assert start_corner[1] < end_corner[1]
    assert isinstance(data_to_draw, Nonacell)
    
    destination_size = (end_corner[0]-start_corner[0], end_corner[1]-start_corner[1])
    subdivisionCorners = [[(start_corner[0]+int(destination_size[0]/3.0*dataX), start_corner[1]+int(destination_size[1]/3.0*dataY)) for dataX in range(4)] for dataY in range(4)]
    
    for driveDataY in range(3):
        for driveDataX in range(3):
            dataForHere = data_to_draw[driveDataY][driveDataX]
            #if driveDataY == driveDataX == 0:
            #    dataForHere = 0
            subStartCorner = subdivisionCorners[driveDataY][driveDataX]
            subEndCorner = subdivisionCorners[driveDataY+1][driveDataX+1]
            if isinstance(dataForHere, int):
                for yIndex, y in enumerate(range(subStartCorner[1], subEndCorner[1])):
                    for xIndex, x in enumerate(range(subStartCorner[0], subEndCorner[0])):
                        #if y%3 == 1 and x%3 == 1:
                        #    dataForHere = (y+x)/4
                        #if y%3 == 2 and x%3 == 2:
                        #    dataForHere = 128
                        #if yIndex == 0:
                        #    dataForHere = 128
                        #if xIndex == 0:
                        #    dataForHere = 256
                        render_target[y][x] = dataForHere
            else:
                assert isinstance(dataForHere, Nonacell)
                rasterize_nonatree(render_target, subStartCorner, subEndCorner, dataForHere)

def make_child_nonatree(parent_pos: complex, parent_size: complex, iter_limit, depth_limit, tree, subX, subY):
    newChild = make_nonatree(screen_to_complex((subX, subY), (3,3), parent_pos, parent_size, centered_sample=True), parent_size/3.0, iter_limit, depth_limit-1, center_value=tree[subY][subX])
    tree[subY][subX] = newChild
    newChild.parent = tree
                
def make_nonatree(tree_pos, tree_size, iter_limit, depth_limit, center_value=None):
    assert isinstance(center_value, int) or center_value is None
    tree = Nonacell(view_pos=tree_pos, view_size=tree_size)
    for x, y, seed in get_seeds((3,3), tree_pos, tree_size, centered_sample=True):
    #for subX, subY in range2d(3,3):
        if (center_value is not None) and x==1 and y==1:
            tree[y][x] = center_value
            continue
        #seed = tree.get_child_pos(subX, subY)
        iters = c_to_mandel_itercount_fast(seed, iter_limit, 4)
        if iters == -1:
            iters = 0
        tree[y][x] = iters
    skipCenter = False
    if sum(sum(row) for row in tree) == 0:
        skipCenter = True
    """
    if depth_limit > 1:
        for x, y, seed in get_seeds((3,3), camera_pos, view_size, centered_sample=True):
            assert isinstance(tree[y][x], int)
            if x == y == 1 and skipCenter:
                tree[y][x] = 1000
                continue
            tree[y][x] = make_nonatree(seed, view_size/3.0, iter_limit, depth_limit-1, center_value=tree[y][x])
            tree[y][x].parent = tree
    """
    if depth_limit > 1:
        for y in range(3):
            for x in range(3):
                make_child_nonatree(tree_pos, tree_size, iter_limit, depth_limit, tree, x, y)
    return tree # <------.
    
def advance_nonatree_frontier(tree, depth_limit):
    if depth_limit <= 1:
        return
    for subY, row in enumerate(tree):
        for subX, child in enumerate(row):
            if isinstance(child, Nonacell):
                advance_nonatree_frontier(child, depth_limit-1)
            else:
                assert isinstance(child, int)
                raise NotImplementedError()
                #make_child_nonatree(
        
def test_nonatree_mandelbrot(camera_pos, view_size, iter_limit, depth_limit):
    screenData = construct_data(screen.get_size(), default_value=0)
    tree = make_nonatree(camera_pos, view_size, iter_limit, depth_limit)
    rasterize_nonatree(screenData, (0,0), screen.get_size(), tree)
    for row in screenData:
        for x in range(len(row)):
            row[x] *= 10
    draw_squished_ints_to_screen([screenData])
        

def test_buddhabrot(camera_pos, view_size, iter_limit, supersampling=1, bidirectional_subsampling=1, count_scale=1):
    if bidirectional_subsampling != 1:
        assert linear_subsampling == 1
    output_name="glitchy_polarsectbrot_RincabsGincrBinci_{}pos{}fov{}itr{}biSuper{}biSub{}count_".format(camera_pos, view_size, iter_limit, supersampling, bidirectional_subsampling, count_scale)
    journeyFun = c_to_mandel_journey
    def specializedDraw():
        draw_squished_ints_to_screen([increasedAbsVisitCountMatrix, increasedRealVisitCountMatrix, increasedImagVisitCountMatrix])
    supersize = (screen.get_size()[0]*supersampling, screen.get_size()[1]*supersampling)
    
    increasedAbsVisitCountMatrix = construct_data(screen.get_size(), default_value=0)
    increasedRealVisitCountMatrix = construct_data(screen.get_size(), default_value=0)
    increasedImagVisitCountMatrix = construct_data(screen.get_size(), default_value=0)
    
    dotCount = 0
    drawnDotCount = 0
    journeyPointCount = 0
    keptJourneyPointCount = 0
    #errorCount = 0
    errorCounter = [0]
    
    def crashlessPolarIntersection(seg0, seg1):
        try:
            return polar_space_segment_intersection(seg0, seg1)
        except AssertionError:
            errorCounter[0] += 1
            return None
    
    for i, x, y, seed in enumerate_flatly(get_seeds(supersize, camera_pos, view_size, centered_sample=False)):
        if x==0 and y%128 == 0:
            #print(testVar)
            specializedDraw()
            if y%128 == 0:
                # print("journey points: {}. kept: {}.".format(journeyPointCount, keptJourneyPointCount))
                screenshot(name_prefix=output_name+"{}of{}rows{}of{}ptskept{}of{}dotsdrawn{}errs_".format(y, supersize[1], keptJourneyPointCount, journeyPointCount, drawnDotCount, dotCount, errorCounter[0]))
        if not (x%bidirectional_subsampling == 0 and y%bidirectional_subsampling == 0):
            continue
        journey = journeyFun(seed)
        constrainedJourney = [item for item in constrain_journey(journey, iter_limit, 4)]
        
        journeyPointCount += len(constrainedJourney)
        if len(constrainedJourney) >= iter_limit:
            continue
        keptJourneyPointCount += len(constrainedJourney)
        
        journeySelfIntersections = gen_intersections(constrainedJourney, intersection_fun=crashlessPolarIntersection)
        #doubleJourneySelfIntersections = gen_intersections(journeySelfIntersections)
        for point in journeySelfIntersections:
            screenPixel = complex_to_screen(point, screen.get_size(), camera_pos, view_size)
            dotCount += 1
            try:
                #visitCounterMatrix[screenPixel[1]][screenPixel[0]] += 1
                if abs(point) > abs(seed):
                    increasedAbsVisitCountMatrix[screenPixel[1]][screenPixel[0]] += count_scale
                if point.real > seed.real:
                    increasedRealVisitCountMatrix[screenPixel[1]][screenPixel[0]] += count_scale
                if point.imag > seed.imag:
                    increasedImagVisitCountMatrix[screenPixel[1]][screenPixel[0]] += count_scale
                drawnDotCount += 1
            except IndexError:
                pass
    specializedDraw()
    screenshot(name_prefix=output_name)



test_abberation([0], 0, 16384)
#test_buddhabrot(-0.5+0j, 3+3j, 64, supersampling=4, bidirectional_subsampling=1, count_scale=1.5) #squish_fun=lambda val: squish_unsigned(val**0.5,255)
#test_nonatree_mandelbrot(-0.5+0j, 4+4j, 64, 6)
stall_pygame()




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