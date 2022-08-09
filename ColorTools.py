import math

from inlinetesting.TestingBasics import assert_single_arg_fun_obeys_dict

CONST_2_OVER_PI = 2.0/math.pi
CONST_510_OVER_PI = 510/math.pi


    
"""
def automatic_color(value):
    if not value >= 0:
        return (0, 0, 0)
    if isinstance(value, float):
        value = int(value)
    #return (min(value, 255), min(int(math.log(max(1, value), 2)), 255), min(int(math.log(max(1, value), 20)), 255))
    return (min((((value+1)/256.0)**0.5)*256, 255), min(value, 255), min(int(math.log(max(1, value), 2)), 255))
"""

"""
def _old_atan_squish_unsigned(input_value, max_output):
    a = input_value / float(max_output)
    a = math.atan(a)/math.pi*2.0
    return a*max_output
"""

def atan_squish_to_float_unsigned_uniform(input_value, max_output):
    trueScale = max_output*CONST_2_OVER_PI
    return math.atan(input_value / trueScale) * trueScale
    
assert 9.98 < atan_squish_to_float_unsigned_uniform(10.0, 256) < 9.99
assert 229.6 < atan_squish_to_float_unsigned_uniform(999.0, 256) < 229.7


def atan_squish_to_byteint_unsigned_uniform_nearest(input_value):
    return round(math.atan(input_value/CONST_510_OVER_PI)*CONST_510_OVER_PI)
    
assert_single_arg_fun_obeys_dict(atan_squish_to_byteint_unsigned_uniform_nearest, dict([(i,i) for i in range(0,32)]+[(64,61), (4096,249), (32768,254), (65536,255), (256*32768*32768*1048576*16*16,255)]))
"""
print("color response preview (appx number of unique output levels for step size setting):")
print("__|".rjust(15, "_") + "-hitCount-->")
for stepSize in ["stepSize", 1,2,4,8,16]:
    print((str(stepSize)+" |").rjust(15, " "), end="")
    for hitCount in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if stepSize == "stepSize":
            print(str(hitCount).rjust(6, "_"), end="")
        else:
            print(str(len(set(atan_squish_to_byteint_unsigned_uniform_nearest(i*stepSize) for i in range(hitCount)))).rjust(6, " "), end="")
    print("")
"""
def preview_inverted(input_function, test_range):
    x = 0
    for printIndex, num in enumerate(test_range):
        while input_function(x) < num:
            x += 1
        print("{}->{}  ".format(x, input_function(x)), end="")
        if printIndex%8 == 0:
            print("")
            
"""
def make_color_bases_in_rgb(input_count):
    # make a cone of vectors starting from the origin and summing to (1.0, 1.0, 1.0).
    raise NotImplementedError()
    for chi in range(3):
        assert sum(result[chi] for result in results) == 1.0
    return results
"""
def lerp_with_waypoints(input_list, t):
    assert len(input_list) >= 2
    assert 0 <= t <= 1
    raise NotImplementedError()

def make_color_bases_in_rgb(input_count):
    raise NotImplementedError()
