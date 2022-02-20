

COMPLEX_ERROR_TOLERANCE = (2**-36)

"""
def assert_equal_or_isinstance(thing0, thing1):
    ...
"""

def assert_equal(thing0, thing1):
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)


def dict_swiss_cheese_access(dict_list, key):
    i = 0
    for i, currentDict in enumerate(dict_list, 1):
        if key in currentDict:
            return currentDict[key]
    raise KeyError("could not find key {} in any of {} dicts.".format(repr(key), i + 1))
    
def dict_swiss_cheese_union(dict_list):
    assert not iter(dict_list) is iter(dict_list)
    keySet = set()
    for currentDict in dict_list:
        keySet.update(set(currentDict.keys()))
    result = dict()
    for key in keySet:
        result[key] = dict_swiss_cheese_access(dict_list, key)
    return result
assert_equal(dict_swiss_cheese_union([{1:2, 2:4, 3:6}, {3:300, 4:400}]), {1:2, 2:4, 3:6, 4:400})


def non_overridably_curry_kwarg_dict(input_fun, kwarg_dict):
    def inner(*args, **kwargs):
        return input_fun(*args, **kwarg_dict, **kwargs)
    return inner

"""
def overridably_curry_kwarg_dict(input_fun, kwarg_dict):
    raise NotImplementedError("not tested! also, for performance, should be done using copy and editing defaults!")
    def inner(*args, **kwargs):
        return input_fun(*args, **(dict_swiss_cheese_union([kwargs, kwarg_dict])))
    return inner
"""


def lpack_exception_raised_by(fun_to_test):
    def inner(*args, **kwargs):
        resultNormalPart, resultExceptionPart = (None, None)
        try:
            resultNormalPart = fun_to_test(*args, **kwargs)
        except Exception as e:
            resultExceptionPart = e
        return (resultExceptionPart, resultNormalPart)
    return inner


def get_exception_raised_by(fun_to_test):
    def inner(*args, **kwargs):
        result = lpack_exception_raised_by(fun_to_test)(*args, **kwargs)[0]
        assert isinstance(result, Exception) or result is None
        return result
    return inner
    
"""
def get_only_non_none_value(input_seq):
    result = None
    for item in input_seq
        if item is not None:
            assert result is None, "could not assure only one non-none value - there was more than one."
            result = item
    assert result is not None, "There were no non-none values."
    return result
"""

def _base_default_to_exception_raised_by(fun_to_test, classify_exception=False):
    def inner(*args, **kwargs):
        packedResult = lpack_exception_raised_by(fun_to_test)(*args, **kwargs)
        if isinstance(packedResult[0], Exception):
            assert packedResult[1] is None
            if classify_exception:
                return type(packedResult[0])
            else:
                return packedResult[0]
        else:
            assert packedResult[0] is None
            return packedResult[1]
    return inner
default_to_exception_raised_by = non_overridably_curry_kwarg_dict(_base_default_to_exception_raised_by, {"classify_exception":False})
default_to_exception_type_raised_by = non_overridably_curry_kwarg_dict(_base_default_to_exception_raised_by, {"classify_exception":True})
assert_equal(default_to_exception_raised_by(int)("5"), 5)
assert isinstance(default_to_exception_raised_by(int)("a"), ValueError)
    
for testArg, desiredType in [([1,2,3], int), (5,TypeError), ([1,"2"], TypeError)]:
    testResult = default_to_exception_raised_by(sum)(testArg)
    if not type(testResult) == desiredType:
        assert False, (testArg, desiredType, testResult)
    
@default_to_exception_type_raised_by
def testValueErrorA(val):
    if val < 0:
        raise ValueError("A: it was negative.")
    else:
        return val
        
assert testValueErrorA(5) == 5
assert testValueErrorA(-1) == ValueError
del testValueErrorA


def assert_single_arg_fun_obeys_dict(fun_to_test, qanda_dict):
    for i, pair in enumerate(qanda_dict.items()):
        testResult = fun_to_test(pair[0])
        assert testResult == pair[1], "failure for test {}, pair={}, testResult={}.".format(i, pair, testResult)
assert_single_arg_fun_obeys_dict(str, {-1:"-1", 5:"5", complex(1,2):"(1+2j)"})
assert default_to_exception_type_raised_by(assert_single_arg_fun_obeys_dict)(int, {"1":2, "3":4}) == AssertionError
assert default_to_exception_type_raised_by(assert_single_arg_fun_obeys_dict)(int, {"a":2, "c":4}) == ValueError


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













def assert_empty(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        return
    assert False, "input seq was not empty, first item was {}.".format(repr(first))
    