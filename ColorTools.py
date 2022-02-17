import math
CONST_2_OVER_PI = 2.0/math.pi
CONST_510_OVER_PI = 510/math.pi


def assert_single_arg_fun_obeys_dict(fun_to_test, qa_dict):
    for i, pair in enumerate(qa_dict.items()):
        testResult = fun_to_test(pair[0])
        assert testResult == pair[1], "failure for test {}, pair={}, testResult={}.".format(i, pair, testResult)
    
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
def squish_color(input_color):
    assert isinstance(input_color, tuple)
    #channel_norm_fun = lambda val: min(int(math.atan(val/256.0*(math.pi/2.0))*(2.0/math.pi)*256.0), 255)
    channel_norm_fun = lambda val: min(255, int(atan_squish_unsigned(val, 256)))
    result = tuple((channel_norm_fun(item) for item in input_color))
    assert all(item >= 0 for item in result)
    assert all(item < 256 for item in result)
    assert all(isinstance(item, int) for item in result)
    assert len(result) == 3
    return result
"""