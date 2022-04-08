import math
import itertools

import os; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import numpy

from PureGenTools import gen_chunks_as_lists
from TestingBasics import print_and_reduce_repetition



def matrix_abs(matA):
    return math.hypot(*matA.flatten().tolist()[0])
    
    
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
    return wrap_to_square_matrix_columns(*args, **kwargs).T


def identity_matrix(side_length, *, scale=1.0):
    return numpy.matrix([([scale*0 for i in range(0,y)] + [scale] + [scale*0 for i in range(y+1,side_length)]) for y in range(side_length)])
    
assert (identity_matrix(1, scale=1) == numpy.matrix('1')).all()
assert (identity_matrix(2, scale=1) == numpy.matrix('1 0; 0 1')).all()
assert (identity_matrix(3, scale=1) == numpy.matrix('1 0 0; 0 1 0; 0 0 1')).all()
assert (identity_matrix(5, scale=3) == numpy.matrix('3 0 0 0 0; 0 3 0 0 0; 0 0 3 0 0; 0 0 0 3 0; 0 0 0 0 3')).all()



_float_factorials = [float(math.factorial(i)) for i in range(0,171)]

def matrix_exp(matA): # , fast=False):
    # termGen = ((matA**n)/_float_factorials[n] for n in range(0,171)) # multiplying out the power one multiplication per loop is about 5-10x faster for 1024x1024 matrices.
    
    result = matA*0.0
    previousResult = matA*0.0
    
    powerOfMatA = identity_matrix(matA.shape[0], scale=1.0)
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
        error = result-previousResul
        errorAbs = matrix_abs(error)
        print("    last error size was {}, or relatively {}.".format(errorAbs, errorAbs/matrix_abs(result)))
    return result
    
assert matrix_eq(matrix_exp(numpy.matrix('0 -1; 1 0')*math.pi), identity_matrix(2, scale=-1), equality_distance=0.0001)