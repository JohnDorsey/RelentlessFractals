import math
    
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

def atan_squish_unsigned_uniform(input_value, max_output):
    trueScale = max_output*2.0/math.pi
    return math.atan(input_value / trueScale) * trueScale
assert 9.98 < atan_squish_unsigned_uniform(10.0, 256) < 9.99
assert 229.6 < atan_squish_unsigned_uniform(999.0, 256) < 229.7
    
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