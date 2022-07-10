import math
import cmath
import itertools
import copy

import os; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import numpy
# NumpyLinAlgError = numpy.linalg.LinAlgError

from PureGenTools import gen_chunks_as_lists, take_first_and_iter
from TestingBasics import print_and_reduce_repetition
from SegmentGeometry import lerp, lerp_confined


def matrix_abs(matA):
    """ abs() is used because math.hypot can't accept complex values. """
    return math.hypot(*[abs(item) for item in matA.flatten().tolist()[0]])
    #return math.hypot(abs(matA).flatten().tolist()
assert matrix_abs(numpy.matrix([[0, 0], [1j, 0]])) == 1.0

def list_abs(input_seq):
    return math.hypot(*[abs(item) for item in input_seq])
    

    
    
def get_normalized_matrix(matA):
    return matA / matrix_abs(matA)
    
    
def matrix_eq(matA, matB, *, equality_distance=None):
    if equality_distance is None:
        return (matA == matB).all()
    else:
        return matrix_abs(matA - matB) < equality_distance
    

assert matrix_eq(numpy.matrix('0 1; 2 3'), numpy.matrix('0 2; 4 6')/2)
assert not matrix_eq(numpy.matrix('0 1; 2 3'), numpy.matrix('5 6; 7 8'))


def matrix_rows_flattened_to_list(matA):
    return matA.flatten().tolist()[0]
def matrix_columns_flattened_to_list(matA):
    return matrix_rows_flattened_to_list(matA.T)


def gen_matrix_rows(matA):
    for y in range(matA.shape[0]):
        yield matA[y].tolist()[0]

def gen_matrix_columns(matA):
    for x in range(matA.shape[1]):
        yield matA[:,x].tolist()[0]



def repeat_to_square_matrix_columns(input_vec):
    result = numpy.matrix([[input_vec[i] for ii in range(len(input_vec))] for i in range(len(input_vec))])
    return result
    
assert matrix_eq(repeat_to_square_matrix_columns([0,1,2]), numpy.matrix('0 0 0; 1 1 1; 2 2 2'))
assert matrix_eq(repeat_to_square_matrix_columns([5]), numpy.matrix('5'))

    
    
def get_containing_square_info(val):
    assert val >= 2
    containingSquareSide = math.ceil((val-0.5)**0.5)
    assert (containingSquareSide-1)**2 < val
    emptySpaceCount = containingSquareSide**2 - val
    assert emptySpaceCount >= 0
    return (containingSquareSide, emptySpaceCount)
    
assert get_containing_square_info(31) == (6,5)
assert get_containing_square_info(36) == (6,0)
assert get_containing_square_info(25) == (5,0)
assert get_containing_square_info(26) == (6,10)
    
"""
def pad_last_to_square(input_2dlist, default):
    assert len(input_2dlist) > 0
    assert 0 < len(input_2dlist[-1]) <= len(input_2dlist)
    missingCount = len(input_2dlist[-1])-len(input_2dlist)
    input_2dlist[-1].extend(itertools.repeat(default, missingCount))
    assert len(input_2dlist[-1]) == len(input_2dlist)
"""
    
def wrap_to_square_matrix_rows(input_list, use_padding=True):
    inputLen = len(input_list)
    if inputLen <= 1:
        raise NotImplementedError("small")
    sideLen, emptySpaceCount = get_containing_square_info(inputLen)
    if use_padding:
        itemGen = itertools.chain(input_list, itertools.repeat(0, emptySpaceCount))
        rowList = list(gen_chunks_as_lists(itemGen, sideLen, allow_partial=False))
        assert len(rowList) == sideLen
        assert len(rowList[-1]) == sideLen
        result = numpy.matrix(rowList)
        assert result.shape == (sideLen, sideLen)
        return result
    else:
        raise NotImplementedError("not allowing padding...")

def wrap_to_square_matrix_columns(*args, **kwargs):
    return wrap_to_square_matrix_rows(*args, **kwargs).T


def identity_matrix(side_length, *, scale=1.0):
    return numpy.matrix([([scale*0 for i in range(0,y)] + [scale] + [scale*0 for i in range(y+1,side_length)]) for y in range(side_length)])
    
assert (identity_matrix(1, scale=1) == numpy.matrix('1')).all()
assert (identity_matrix(2, scale=1) == numpy.matrix('1 0; 0 1')).all()
assert (identity_matrix(3, scale=1) == numpy.matrix('1 0 0; 0 1 0; 0 0 1')).all()
assert (identity_matrix(5, scale=3) == numpy.matrix('3 0 0 0 0; 0 3 0 0 0; 0 0 3 0 0; 0 0 0 3 0; 0 0 0 0 3')).all()


def get_square_matrix_diagonal(matA):
    assert matA.shape[0] == matA.shape[1]
    return [matA[i, i] for i in range(matA.shape[0])]
    
def get_square_matrix_opposite_diagonal(matA):
    assert matA.shape[0] == matA.shape[1]
    sideLen = matA.shape[0]
    return [matA[i, sideLen-1-i] for i in range(sideLen)]
    
    
def get_row_sums(matA):
    return [sum(row) for row in gen_matrix_rows(matA)]





    
def evaluate_series_until_zero(term_seq):
    readOnlyTerm, termGen = take_first_and_iter(term_seq)
    result = copy.deepcopy(readOnlyTerm)
    
    for readOnlyTerm in termGen:
        result += readOnlyTerm
        isZero = (not readOnlyTerm.any()) if isinstance(readOnlyTerm, numpy.matrix) else (abs(readOnlyTerm) == 0)
        if isZero:
            return result
    assert False, "ran out of items?"
    
    

_float_factorials = [float(math.factorial(i)) for i in range(0,171)]

def matrix_exp_series_term_gen(matA):
    term = matA**0
    # termCopy = matA**0
    yield term
    for n in itertools.count(1):
        term *= matA
        term /= n
        # numpy.copyto(dst=termCopy, src=term)
        yield term
    assert False
    
    
"""

old:
    result = identity_matrix(matA.shape[0], scale=dtypeOne*0)
    previousResult = identity_matrix(matA.shape[0], scale=dtypeOne*0)
    powerOfMatA = identity_matrix(matA.shape[0], scale=dtypeOne)
    for n in range(0, 171):
        if n > 0:
            powerOfMatA *= matA
        seriesTerm = powerOfMatA/_float_factorials[n]
        
        numpy.copyto(dst=previousResult, src=result)
        result += seriesTerm
        if matrix_eq(previousResult, result):
            # print("took {} iters.".format(i))
            return result
    if print_and_reduce_repetition("matrix_exp: WARNING: ran out of precision for factorial! inaccurate result!"):
        error = result-previousResult
        errorAbs = matrix_abs(error)
        print("    last error size was {}, or relatively {}.".format(errorAbs, errorAbs/matrix_abs(result)))
    return result
    
    
def matrix_exp(matA, *, dtype=complex): # , fast=False):
    # termGen = ((matA**n)/_float_factorials[n] for n in range(0,171)) # multiplying out the power one multiplication per loop is about 5-10x faster for 1024x1024 matrices.
    if dtype in (complex, float, int):
        dtypeOne = dtype("1")
    else:
        raise TypeError(dtype)
    assert matA.dtype == dtype, (matA[0:5,0:5], type(matA[0,0]), matA.dtype)
    
    
    result = matA*0
    assert hasattr(matA, "__any__")
    return evaluate_series_until_zero(matrix_exp_series_term_gen(matA))
"""

def matrix_exp(matA):
    return evaluate_series_until_zero(matrix_exp_series_term_gen(matA))
    
assert matrix_eq(matrix_exp(numpy.matrix('0 -1; 1 0')*math.pi), identity_matrix(2, scale=-1), equality_distance=0.0001)



def matrix_sin_series_term_gen(matA):
    # https://www.mathsisfun.com/algebra/taylor-series.html
    # x - x**3/3! + x**5/5! - ...
    # a_n = ((-1)**n / (2n+1)!)*(x**(2n+1))
    # a_n = a_(n-1) * (-1) / (2*n + 1) / (2*n) * (x**2)
    term = copy.deepcopy(matA)
    yield term
    matASquared = matA**2
    for n in itertools.count(1):
        term *= matASquared
        term *= -1 / ((2*n+1) * (2*n))
        yield term
    assert False
    
def matrix_sin(matA):
    return evaluate_series_until_zero(matrix_sin_series_term_gen(matA))
    

def matrix_cos_series_term_gen(matA):
    # 1 - x**2/2! + x**4/4!
    # a_n = a_(n-1) / 2n / 2n-1 * x**2
    term = matA**0
    yield term
    matASquared = matA**2
    for n in itertools.count(1):
        term *= matASquared
        term *= -1 / ((2*n) * (2*n - 1))
        yield term
    assert False
        
def matrix_cos(matA):
    return evaluate_series_until_zero(matrix_cos_series_term_gen(matA))



class SingularMatrixInversionError(Exception):
    pass

class OtherMatrixInversionError(Exception):
    pass

def matrix_inv(matA):
    """
    don't use matA.I, as it does not show errors when the matrix is not invertable.
    """
    try:
        result = numpy.linalg.inv(matA)
    except numpy.linalg.LinAlgError as lae:
        if lae.args == ('Singular matrix',):
            raise SingularMatrixInversionError(*lae.args)
        else:
            raise OtherMatrixInversionError(*lae.args)
    return result


def two_listvector_angle_shortest(lv0, lv1):
    # https://www.cuemath.com/geometry/angle-between-vectors/
    # cos(theta) = (a dot b)/(|a|*|b|)
    # sin(theta) = (a cross b)/(|a|*|b|)
    shorterLength = min(len(lv0), len(lv1))
    if shorterLength == 0:
        raise ValueError("can't work with vectors of length zero (lengths {} and {}).".format(len(lv0), len(lv1)))
    cosVal = numpy.dot(lv0[:shorterLength], lv1[:shorterLength])/(list_abs(lv0)*list_abs(lv1))
    return cmath.acos(cosVal)
assert two_listvector_angle_shortest([0,1,0],[1,0,0]) == math.pi/2
assert two_listvector_angle_shortest([0,1,0],[0,-1,0]) == math.pi






def generic_abs(value):
    if isinstance(value, numpy.matrix):
        return matrix_abs(value)
    elif isinstance(value, (tuple, list)):
        return list_abs(value)
    else:
        return abs(value)
"""
def interleave(*args):
    inputGens = [iter(arg) for arg in args]
    while True:
        for currentGenIndex, currentGen in enumerate(inputGens):
        nextItem =
"""

def stagger_out_range(start, low_stop, high_stop):
    if start is None:
        assert low_stop + 1 < high_stop
        start = low_stop + 1
    assert low_stop < start < high_stop
    yield start
    for distance in itertools.count(1):
        high, low = (start+distance, start-distance)
        if high < high_stop:
            yield high
        if low > low_stop:
            yield low
        if not (high < high_stop or low > low_stop):
            return
    assert False
    
assert sorted(list(stagger_out_range(7,5,10))) == list(range(6,10))

"""
def minimize_bounded(fun, a, b, eq_bias=-1):
    assert a < b
    assert eq_bias in {-1, 0, 1}
    bundleFun = lambda x: (x, fun(x))
    # midpointOf = lambda x0, x1: (x0+x1)*0.5
    # current = (bundleFun(x) for x in (a, m, b))
    # current = tuple((x,fun(x)) for x in (a,(a+b)*0.5,b))
    m = (a+b)*0.5
    aBun, mBun, bBun = tuple(bundleFun(x) for x in (a,m,b))
    if aBun[1] == bBun[1]:
        choice = eq_bias
    elif aBun[1] < bBun[1]:
        choice = -1
    else:
        assert aBun[1] > bBun[1]:
        choice = 1
    if choice == 0:
        raise ValueError("???")
    else:
        if choice == -1:
            bBun, mBun = (mBun, None)
        else:
            assert choice == 1:
            aBun, mBun = (mBun, None)
        m = (a+b)*0.5
        mBun = (m, fun(m))
"""

def _auto_inverse(fun, value, equality_distance=2**-32):

    currentBest = (value, fun(value))
    currentIdealSearchDistanceInv = None
    
    def genValuesToTest(val, fVal, startSearchDistanceInv=None):
        for searchDistanceInv in stagger_out_range(startSearchDistanceInv or 2, 1, 256**2):
            searchDistance = 1.0/searchDistanceInv
            for testValue in [(1.0-searchDistance)*val, (1.0+searchDistance)*val, lerp_confined(val, fVal, searchDistance), lerp(val, fVal, -searchDistance)]:
                yield (searchDistanceInv, testValue)
    
    for i in range(65536**2):
        if generic_abs(value-currentBest[1]) < equality_distance:
            return currentBest[0]
        for testIndex, (searchDistanceInv, _testValue) in enumerate(genValuesToTest(currentBest[0], currentBest[1], startSearchDistanceInv=currentIdealSearchDistanceInv)):
            currentTest = (_testValue, fun(_testValue))
            if generic_abs(value - currentTest[1]) < generic_abs(value - currentBest[1]):
                print("in step {}, test {} was successful. searchDistanceInv={}, ideal is {}.".format(i, testIndex, searchDistanceInv, currentIdealSearchDistanceInv))
                currentIdealSearchDistanceInv = searchDistanceInv
                currentBest = currentTest
                break
        else:
            print("convergence stopped on step {} at {}.".format(i, currentTest))
            return
    print("convergence took too long.")




