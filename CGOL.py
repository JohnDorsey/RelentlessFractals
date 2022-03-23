import itertools
from enum import Enum
import random
import numbers


from TestingAtoms import assert_equal, assure_isinstance
from PureGenTools import izip_uniform, gen_track_recent_full






class EdgeMode(Enum):
    WRAP = "wrap"
    VOID = "void"
    SHRINK = "shrink"
    

def cgol_get_stepped_center_cell(input_matrix):
    ringSum = sum(input_matrix[0]) + sum(input_matrix[2]) + input_matrix[1][0] + input_matrix[1][2]
    assert input_matrix[1][1] is not None
    if ringSum < 2 or ringSum > 3:
        return 0
    else:
        if ringSum == 3:
            return 1
        else:
            assert ringSum == 2
            return input_matrix[1][1]
    

def _cgol_get_stepped_middle_row_shrinkx(input_row_list):
    assert len(input_row_list) == 3
    windowDataGen = izip_uniform(*[gen_track_recent_full(row, count=3) for row in input_row_list])
    return [cgol_get_stepped_center_cell(windowData) for windowData in windowDataGen]


def _gen_extend_list_by_wrapping(input_list, left_length=None, right_length=None):
    assert left_length >= 0 and left_length <= len(input_list)
    assert right_length >= 0 and right_length <= len(input_list)
    return itertools.chain(input_list[-left_length:], input_list, input_list[0:right_length])

assert list(_gen_extend_list_by_wrapping([5,6,7,8], left_length=1, right_length=2)) == [8,5,6,7,8,5,6]





def get_zero_duplicate(data, count=1):
    if isinstance(data, numbers.Number):
        result = [data*0 for i in range(count)]
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], numbers.Number):
            result = [type(data)(data[0]*0 for i in range(len(data))) for ii in range(count)]
        else:
            raise TypeError("bad type or too deep to efficiently convert: {}.".format(type(data[0])))
    else:
        raise TypeError("bad type {}.".format(type(data)))
        
    if count == 1:
        return result[0]
    else:
        return result


def with_edge_mode_adjustment(input_data, edge_mode, allow_storage=False):
    if edge_mode is EdgeMode.SHRINK:
        return input_data
    else:
        if hasattr(input_data, "__getitem__"):
            inputDataList = input_data
        else:
            if allow_storage:
                inputDataList = list(input_data)
            else:
                raise TypeError("input_data must have __getitem__ when edge_mode is WRAP and allow_storage is False.")
        
        if edge_mode is EdgeMode.WRAP:
            return _gen_extend_list_by_wrapping(inputDataList, left_length=1, right_length=1)
        else:
            assert edge_mode is EdgeMode.VOID, "bad edge mode: {}.".format(repr(edge_mode))
            # assert type(inputDataList[0]) == type(inputDataList[-1])
            lDupe, rDupe = get_zero_duplicate(inputDataList[0], count=2)
            return itertools.chain([lDupe], inputDataList, [rDupe])
        

"""
def with_edge_mode_adjustments(input_data, x_edge_mode=None, y_edge_mode=None):
    workingData = with_edge_mode_adjustment(input_data, y_edge_mode)
    result = (with_edge_mode_adjustment(row, x_edge_mode) for row in workingData)
    return result
"""

def cgol_get_stepped_middle_row(input_row_seq, x_edge_mode=None):
    assert hasattr(input_row_seq, "__getitem__"), "dropped data is possible."
    workingInputRows = [with_edge_mode_adjustment(assure_isinstance(inputRow, (list, tuple)), x_edge_mode) for inputRow in input_row_seq]
    assert len(workingInputRows) == 3
    return _cgol_get_stepped_middle_row_shrinkx(workingInputRows)


def _cgol_gen_stepped_rows_shrinky(input_row_seq, x_edge_mode=None):
    for currentRowTriplet in gen_track_recent_full(input_row_seq, count=3):
        assert iter(currentRowTriplet[2]) is not iter(currentRowTriplet[2])
        yield cgol_get_stepped_middle_row(currentRowTriplet, x_edge_mode=x_edge_mode)
        

def cgol_gen_stepped_rows(input_row_seq, x_edge_mode=None, y_edge_mode=None):
    workingInputRows = with_edge_mode_adjustment(input_row_seq, y_edge_mode)
    # workingInputRowsB = (with_edge_mode_adjustment(inputRow, x_edge_mode) for inputRow in workingInputRowsA)
    return _cgol_gen_stepped_rows_shrinky(workingInputRows, x_edge_mode=x_edge_mode)
    
test0 = [[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,1,1,0,0],[0,1,0,0,0,1,1,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
test1 = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,0,0],[1,1,1,0,0,1,1,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
assert_equal(list(cgol_gen_stepped_rows(test0, x_edge_mode=EdgeMode.WRAP, y_edge_mode=EdgeMode.WRAP)), test1)
assert_equal(list(cgol_gen_stepped_rows(test1, x_edge_mode=EdgeMode.WRAP, y_edge_mode=EdgeMode.WRAP)), test0)
del test0, test1
test2 = [[0,1,0],[0,1,0],[0,1,0]]
test3 = [[0,0,0],[1,1,1],[0,0,0]]
assert_equal(list(cgol_gen_stepped_rows(test2, x_edge_mode=EdgeMode.VOID, y_edge_mode=EdgeMode.VOID)), test3)
assert_equal(list(cgol_gen_stepped_rows(test3, x_edge_mode=EdgeMode.VOID, y_edge_mode=EdgeMode.VOID)), test2)
del test2, test3
    
"""
def cgol_gen_stepped_rows_wrapx_shrinky(input_row_seq):
    return (cgol_get_stepped_middle_row_wrapx(currentRowTriplet) for currentRowTriplet in gen_track_recent_full(input_row_seq, count=3))
    
    

def cgol_gen_stepped_rows_wrapx_wrapy(...):
    ...
"""
def demo(size=(48,24), steps=8, p=0.1):
    assert size[0] > 5
    data = [[int(random.uniform(0.0,1.0)<p) for x in range(size[0])] for y in range(size[1])]
    for i in range(steps):
        data = list(cgol_gen_stepped_rows(data, x_edge_mode=EdgeMode.WRAP, y_edge_mode=EdgeMode.WRAP))
        print("/-"+("-"*(size[0]-4))+"-\\")
        print("]\n[".join("".join(" #"[val] for val in row) for row in data))
        print("\\-"+("-"*(size[0]-4))+"-/")



