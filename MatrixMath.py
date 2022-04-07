import math
import itertools

import numpy as np

def make_square_matrix_from_column(input_vec):
    result = np.matrix([[input_vec[i] for ii in range(len(input_vec))] for i in range(len(input_vec))])
    return result

def matrix_abs(matA):
    return math.hypot(*matA.flatten().tolist()[0])
    
def get_normalized_matrix(matA):
    return matA / matrix_abs(matA)
    
def matrix_eq(matA, matB):
    return (matA == matB).all()

# _float_factorials = [float(math.factorial(i)) for i in range

def matrix_exp(matA):
    termGen = ((matA**n)/float(math.factorial(n)) for n in range(0,171))
    result = matA*0.0
    previousResult = None
    for i, term in enumerate(termGen):
        previousResult = result.copy()
        result += term
        if matrix_eq(previousResult, result):
            # print("took {} iters.".format(i))
            return result
    print("matrix_exp: WARNING: ran out of precision for factorial! inaccurate result!")
    return result
        