
#import pathlib
import functools

from TestingAtoms import assert_equal, assert_less, assert_isinstance, AssuranceError, summon_cactus


COMPLEX_ERROR_TOLERANCE = (2**-36)


"""
def assert_equal_or_isinstance(thing0, thing1):
    ...
"""

def floor_binary_round_int(value):
    assert isinstance(value, int), type(value)
    assert value > 0
    return 2**(value.bit_length()-1)

def is_binary_round_int(value):
    return value == floor_round_binary_int(value)

def print_and_reduce_repetition(text, details="", _info=[None, 1]):
    # text = str(thing)
    
    if _info[0] == text:
        _info[1] += 1
    else:
        _info[0] = text
        _info[1] = 1
        
    if _info[1] <= 3 or is_binary_round_int(_info[1]):
        print("{} (now repeated x{}). {}".format(str(_info[0]), _info[1], " details: {}".format(details) if details else ""))
        return True
    else:
        return False










"""

def _dict_swiss_cheese_access(dict_list, key):
    i = 0
    for i, currentDict in enumerate(dict_list, 1):
        if key in currentDict:
            return currentDict[key]
    raise KeyError("could not find key {} in any of {} dicts.".format(repr(key), i + 1))
    
assert _dict_swiss_cheese_access([{1:2, 3:4, 5:6, 9:10}, {5:60, 7:80}], 5) == 6
assert _dict_swiss_cheese_access([{1:2, 3:4, 5:6, 9:10}, {5:60, 7:80}], 7) == 80
assert _dict_swiss_cheese_access([{1:2, 3:4, 5:6, 9:10}, {5:60, 7:80}], 9) == 10

    
def _dict_swiss_cheese_union(dict_list):
    assert not iter(dict_list) is iter(dict_list)
    keySet = set()
    for currentDict in dict_list:
        keySet.update(set(currentDict.keys()))
    result = dict()
    for key in keySet:
        result[key] = _dict_swiss_cheese_access(dict_list, key)
    return result
    
assert_equal(_dict_swiss_cheese_union([{1:2, 2:4, 3:6}, {3:300, 4:400}]), {1:2, 2:4, 3:6, 4:400})


def _non_overridably_curry_kwarg_dict(input_fun, kwarg_dict):
    # raise NotImplementedError("not tested! also, this is a wheel reinvention of builtin partials!")
    def inner(*args, **kwargs):
        return input_fun(*args, **kwarg_dict, **kwargs)
    return inner
print("tests needed for _non_overridably_curry_kwarg_dict")
"""


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
    
assert lpack_exception_raised_by(sum)([3,4,5]) == (None, 12)
testResult = lpack_exception_raised_by(sum)([3,4,"a",5])
assert testResult[1] is None
assert isinstance(testResult[0], TypeError)
del testResult

"""
def lpack_exception_type_raised_by(fun_to_test):
    def inner(*args, **kwargs):
"""
def get_value_returned_or_exception_raised_by(fun_to_wrap):
    def inner(*args, **kwargs):
        exceptionResult = None
        try:
            valueResult = fun_to_wrap(*args, **kwargs)
        except Exception as e:
            if isinstance(e, AssertionError):
                print("TestingBasics.get_value_returned_or_exception_raised_by: warning: suppressing an AssertionError.")
            return e
        return valueResult
    return inner
        


def get_exception_or_none_raised_by(fun_to_test):
    # print("TestingBasics.get_exception_raised_by: todo: maybe should be changed to have more obvious behavior of failing when no exception is raised.")
    def inner(*args, **kwargs):
        result = lpack_exception_raised_by(fun_to_test)(*args, **kwargs)[0]
        assert isinstance(result, Exception) or result is None
        return result
    return inner
    
# test later.
    
    
def raises_instanceof(fun_to_test, exception_types, debug=False):
    def raises_instanceof_inner(*args, **kwargs):
        exceptionResult = get_exception_or_none_raised_by(fun_to_test)(*args, **kwargs)
        result = isinstance(exceptionResult, exception_types)
        if debug and not result:
            print("raises_instanceof: actually got exception {}, not of type {}.".format(repr(exceptionResult), repr(exception_types)))
        return result
    return raises_instanceof_inner
    
def testRaiseIndexError(*args):
    raise IndexError()
assert isinstance(get_exception_or_none_raised_by(testRaiseIndexError)(1,2,3), IndexError)
assert isinstance(get_exception_or_none_raised_by(str)(1), type(None))
assert raises_instanceof(testRaiseIndexError, IndexError)(1,2,3) == True
assert raises_instanceof(str, IndexError)(1) == False
del testRaiseIndexError


def assure_raises_instanceof(fun_to_test, exception_types):
    def assure_raises_instanceof_inner(*args, **kwargs):
        resultingException, resultingValue = lpack_exception_raised_by(fun_to_test)(*args, **kwargs)
        assert isinstance(resultingException, exception_types), f"assure_raises_instanceof: the wrapped function {fun_to_test.__name__} was expected to raise an instance of {exception_types}, but instead raised {repr(resultingException)=} of type {type(resultingException)} (and/or returned {resultingValue})."
        return resultingException
    return assure_raises_instanceof_inner

"""
def assert_raises_instanceof(fun_to_test, exception_types, debug=False):
    print("TestingBasics: warning: assert_raises_instanceof is deprecated. use assure_raises_instanceof, which performs the same test but returns the caught exception.")
    if str(pathlib.Path.cwd()).split("/")[-1] == "Battleship":
        assert False, "fix now"
    else:
        print("TestingBasics: leftover code tests whether this project is Battleship. cleanup needed.")
    if debug:
        raise NotImplementedError("can't debug.")
    def assert_raises_instanceof_inner(*args, **kwargs):
        resultingException, resultingValue = lpack_exception_raised_by(fun_to_test)(*args, **kwargs)
        assert isinstance(resultingException, exception_types), "assert_raises_instanceof: the wrapped function {} was expected to raise {}, but instead raised (exception={}, value={}).".format(fun_to_test.__name__, exception_types, repr(resultingException), resultingValue)
    return assert_raises_instanceof_inner
"""


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

        
def assure_returns_instanceof(desired_type):
    def assure_returns_instanceof__inner_decorator(input_fun):
        def assure_returns_instanceof__inner_fun(*args, **kwargs):
            result = input_fun(*args, **kwargs)
            assert_isinstance(result, desired_type)
            return result
        return assure_returns_instanceof__inner_fun
    return assure_returns_instanceof__inner_decorator











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
default_to_exception_raised_by = functools.partial(_base_default_to_exception_raised_by, classify_exception=False)
default_to_exception_type_raised_by = functools.partial(_base_default_to_exception_raised_by, classify_exception=True)

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


def assert_single_arg_fun_obeys_dict(fun_to_test, q_and_a_dict):
    for i, pair in enumerate(q_and_a_dict.items()):
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
    if isinstance(thing0, (complex,float,int)) and isinstance(thing1, (complex,float,int)):
        result = test_complex_nearly_equal(thing0, thing1, error_tolerance=error_tolerance)
        if debug and not result:
            print(head + "failed in br0.")
        return result
    elif any(isinstance(thing0, testEnterable) and isinstance(thing1, testEnterable) for testEnterable in (tuple, list)):
        if len(thing0) != len(thing1):
            if debug:
                print(head + "lengths differ.")
            return False
        result = all(test_nearly_equal(thing0[i], thing1[i], error_tolerance=error_tolerance, debug=debug) for i in range(max(len(thing0), len(thing1))))
        if debug and not result:
            print(head + "failed in br1.")
        return result
    else:
        result = (thing0 == thing1)
        if debug and not result:
            print(head + "failed in br2.")
        return result


def assert_nearly_equal(thing0, thing1, error_tolerance=COMPLEX_ERROR_TOLERANCE):
    assert test_nearly_equal(thing0, thing1, error_tolerance=error_tolerance, debug=True), "{} does not nearly equal {} with error tolerance {}.".format(repr(thing0), repr(thing1), repr(error_tolerance))













    