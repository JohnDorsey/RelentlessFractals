
#import pathlib
import functools

from inlinetesting.TestingAtoms import assert_equal, assert_less, assert_isinstance, AssuranceError, summon_cactus, AlternativeAssertionError, MeaninglessError, raise_inline



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
        if not isinstance(resultingException, exception_types):
            raise AssuranceError(f"assure_raises_instanceof:\n    the wrapped function {fun_to_test.__name__} was expected to raise an instance of {exception_types},\n    but instead raised {repr(resultingException)=}\n    of type {type(resultingException)}\n    (and/or returned {resultingValue}).")
        return resultingException
    return assure_raises_instanceof_inner

class AssertRaisesInstanceof:
    def __init__(self, exception_types):
        if not (issubclass(exception_types, Exception) or (isinstance(exception_types, tuple) and all(issubclass(item, Exception) for item in exception_types))):
            raise AlternativeAssertionError(f"invalid Exceptions specified: {exception_types}.")
        self.exception_types = exception_types
        
    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_value, traceback_):
        if exc_type is None:
            # there is no exception to cancel. but one was expected, so raise an AssertionError.
            raise AssertionError(f"AssertRaisesInstanceof required an exception of type {self.exception_types}, but did not receive any exception.")
        assert isinstance(exc_type, type), (exc_type, exc_value, traceback_)
        if issubclass(exc_type, self.exception_types):
            return True # cancel the exception by returning something other than None.
        else:
            raise AssertionError(f"AssertRaisesInstanceof required an exception of type {self.exception_types}, but received one of type {exc_type}. {exc_value=}.")
def _raiseValueError():
    raise ValueError("test text")
with AssertRaisesInstanceof(ValueError):
    _raiseValueError()
def _testWrongError():
    with AssertRaisesInstanceof(IndexError):
        _raiseValueError()
assure_raises_instanceof(_testWrongError, AssertionError)()


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




    

def translate_exception_type_inline(fun_to_wrap, translation_dict):
    if not callable(fun_to_wrap):
        raise AlternativeError("fun_to_wrap not callable.")
    if not isinstance(translation_dict, dict):
        raise AlternativeError("invalid translation_dict type.")
    def inner(*args, **kwargs):
        try:
            result = fun_to_wrap(*args, **kwargs)
        except Exception as e:
            for key, value in translation_dict.items():
                if isinstance(e, key):
                    raise value(*e.args) from None
            raise e from None
        return result
    return inner

# tests for raise_inline, which is in TestingAtoms.
assure_raises_instanceof(raise_inline, MeaninglessError)(MeaninglessError, "hello")
assure_raises_instanceof(raise_inline, MeaninglessError)(MeaninglessError("hello"))

#tests for translate_exception_type_inline
assure_raises_instanceof(translate_exception_type_inline(raise_inline, {OverflowError:ZeroDivisionError}), ZeroDivisionError)(OverflowError, "hello")
assure_raises_instanceof(translate_exception_type_inline(raise_inline, {OverflowError:ZeroDivisionError}), ZeroDivisionError)(ZeroDivisionError, "hello")
assure_raises_instanceof(translate_exception_type_inline(raise_inline, {OverflowError:ZeroDivisionError}), MeaninglessError)(MeaninglessError, "hello")
assure_raises_instanceof(translate_exception_type_inline(raise_inline, {(MeaninglessError, OverflowError):ZeroDivisionError}), ZeroDivisionError)(MeaninglessError, "hello")


























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



"""
already exists and is called AssertRaisesInstanceof
class AssertRaises:
    def __init__(self, exception_types):
        if (isinstance(exception_types, Exception) or (isinstance(exception_types, tuple) and all(isinstance(item, Exception) for item in exception_types))):
            raise AlternativeAssertionError()
        self.exception_types = exception_types
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, tb):
        if not issubclass(exc_type, self.exception_types):
            raise AssertionError("Expected {} to be raised, but {} was raised: {}.".format(self.exception_types, exc_type, exc_value)) from None
        else:
            return True

with AssertRaises(ValueError):
    raise ValueError("test")
try:
    with AssertRaises(ValueError):
        raise TypeError("test")
except AssertionError:
    pass
else:
    raise AlternativeAssertionError("AssertRaises failed.")
"""









    