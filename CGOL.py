from PureGenTools import izip_longest, gen_track_recent_full
from enum import Enum



class EdgeMode(Enum):
    WRAP = "wrap"
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


def with_edge_mode_adjustment(input_data, edge_mode):
    if edge_mode is EdgeMode.WRAP:
        if not hasattr(input_data, __getitem__):
            raise TypeError("input data must have __getitem__ when edge mode is WRAP.")
        return _gen_extend_list_by_wrapping(input_data, left_length=1, right_length=1)
    else:
        assert edge_mode is EdgeMode.SHRINK, "bad edge mode."
        return input_data


def with_edge_mode_adjustments(input_data, x_edge_mode=None, y_edge_mode=None):
    workingData = with_edge_mode_adjustment(input_data, y_edge_mode)
    result = (with_edge_mode_adjustments(row, x_edge_mode) for row in workingData)
    return result


def cgol_get_stepped_middle_row(input_row_seq, x_edge_mode=None):
    workingInputRows = [with_edge_mode_adjustment(inputRow, x_edge_mode) for inputRow in input_row_seq]
    assert len(workingInputRows) == 3
    return _cgol_get_stepped_middle_row_shrinkx(workingInputRows)


def _cgol_gen_stepped_rows_shrinky(input_row_seq, x_edge_mode=None):
    return (cgol_get_stepped_middle_row(currentRowTriplet, x_edge_mode=x_edge_mode) for currentRowTriplet in gen_track_recent_full(input_row_seq, count=3))
        
def cgol_gen_stepped_rows(input_row_seq,
"""
def cgol_gen_stepped_rows_wrapx_shrinky(input_row_seq):
    return (cgol_get_stepped_middle_row_wrapx(currentRowTriplet) for currentRowTriplet in gen_track_recent_full(input_row_seq, count=3))
    
    

def cgol_gen_stepped_rows_wrapx_wrapy(...):
    ...
"""

