
import itertools

from TestingAtoms import assert_equal, summon_cactus
from TestingBasics import assure_raises_instanceof
from PureGenTools import izip_uniform_containers



def apply_slice_chain(data, slice_chain):
    for currentSlice in slice_chain:
        data = data[currentSlice]
    return data

def higher_range_linear(descriptions, *, post_slices=None):
    # this might be a reinvention of itertools.product.
    assert len(descriptions) > 0
    ranges = [range(*description) for description in descriptions]
    if post_slices is not None:
        for i, (currentRange, currentSlices) in enumerate(izip_uniform_containers(ranges, post_slices)):
            if currentSlices is None:
                continue
            else:
                if isinstance(currentSlices, tuple):
                    currentSliceChainTup = currentSlices
                else:
                    assert isinstance(currentSlices, slice), "at axis {}, slice chain must be single slice, tuple of slices, or None (got type {}).".format(i, type(currentSlices))
                    currentSliceChainTup = (currentSlices,)
                ranges[i] = apply_slice_chain(currentRange, currentSliceChainTup)
    return itertools.product(*ranges)
                
assert_equal(list(higher_range_linear([(2,5), (3,10,3)])), [(a,b) for a in range(2,5) for b in range(3,10,3)], [(2,3), (2,6), (2,9), (3,3), (3,6), (3,9), (4,3), (4,6), (4,9)])


def list_in_new_order(data, new_order, *, reverse_output=False, _uninitialized=summon_cactus("error_in__list_in_new_order")):
    """ new_order[a] gives b such that output_data[b]==input_data[a]. """
    assert len(new_order) == len(data)
    #assert sorted(new_order) == list(range(len(new_order)))
    reorderedData = [_uninitialized for i in range(len(data))]
    for srcIndex in range(len(new_order)):
        destIndex = (-1-new_order[srcIndex] if reverse_output else new_order[srcIndex])
        assert reorderedData[destIndex] is _uninitialized
        reorderedData[destIndex] = data[srcIndex]
    assert len(reorderedData) == len(new_order)
    assert _uninitialized not in reorderedData
    return reorderedData
    
assert_equal(list_in_new_order([10,20,30,40,50], [1,4,0,2,3]), [30, 10, 40, 50, 20])
assert_equal(list_in_new_order([10,20,30,40,50], [1,4,0,2,3], reverse_output=True), [30, 10, 40, 50, 20][::-1])

def inverse_list_in_new_order(data, source_indices, reverse_output=False, _uninitialized=summon_cactus("error_in__inverse_list_in_new_order")):
    assert len(source_indices) == len(data)
    return [data[source_indices[i]] for i in (range(len(data)-1,-1,-1) if reverse_output else range(0,len(data)))]

assert_equal(inverse_list_in_new_order([10,20,30,40,50], [1,4,0,2,3]), [20, 50, 10, 30, 40])
assert_equal(inverse_list_in_new_order([10,20,30,40,50], [1,4,0,2,3], reverse_output=True), [40, 30, 10, 50, 20])


def higher_range(descriptions, *, post_slices=None, iteration_order=None):
    if iteration_order is not None:
        reorderedDescriptions = list_in_new_order(descriptions, iteration_order, reverse_output=True)
        if post_slices is not None:
            reorderedPostSlices = list_in_new_order(post_slices, iteration_order, reverse_output=True)
        else:
            reorderedPostSlices = None
            
        for unorderedItem in higher_range_linear(reorderedDescriptions, post_slices=reorderedPostSlices):
            reorderedItem = tuple(unorderedItem[-1-srcIndex] for srcIndex in iteration_order)
            yield reorderedItem
    else:
        for item in higher_range_linear(descriptions, post_slices=post_slices):
            yield item

assert_equal(list(higher_range([(2,5), (3,10,3)])), [(a,b) for a in range(2,5) for b in range(3,10,3)], [(2,3), (2,6), (2,9), (3,3), (3,6), (3,9), (4,3), (4,6), (4,9)])
assert_equal(list(higher_range([(2,5), (3,10,3)], iteration_order=[1,0])), [(a,b) for a in range(2,5) for b in range(3,10,3)], [(2,3), (2,6), (2,9), (3,3), (3,6), (3,9), (4,3), (4,6), (4,9)])
assert_equal(list(higher_range([(2,5), (3,10,3)], iteration_order=[0,1])), [(a,b) for b in range(3,10,3) for a in range(2,5)], [(2,3), (3,3), (4,3), (2,6), (3,6), (4,6), (2,9), (3,9), (4,9)])


assert_equal(list(higher_range_linear([(0,2), (33,35), (777,779)],                 )), [(0,33,777),(0,33,778),(0,34,777),(0,34,778),(1,33,777),(1,33,778),(1,34,777),(1,34,778)])
assert_equal(list(higher_range([(0,2), (33,35), (777,779)],                        )), [(0,33,777),(0,33,778),(0,34,777),(0,34,778),(1,33,777),(1,33,778),(1,34,777),(1,34,778)])
assert_equal(list(higher_range([(0,2), (33,35), (777,779)], iteration_order=[2,1,0])), [(0,33,777),(0,33,778),(0,34,777),(0,34,778),(1,33,777),(1,33,778),(1,34,777),(1,34,778)])
assert_equal(list(higher_range([(0,2), (33,35), (777,779)], iteration_order=[1,2,0])), [(0,33,777),(0,33,778),(1,33,777),(1,33,778),(0,34,777),(0,34,778),(1,34,777),(1,34,778)])
            
assert_equal(list(higher_range([(2,5), (3,10,3), (4,)])), [(a,b,c) for a in range(2,5) for b in range(3,10,3) for c in range(4)])
assert_equal(list(higher_range([(2,5), (3,10,3), (4,)], iteration_order=[2,0,1])), [(a,b,c) for a in range(2,5) for c in range(4) for b in range(3,10,3)])

assert_equal(list(higher_range([(2,4), (20,55,5)], post_slices=[None, slice(None,None,3)])), [(2,20),(2,35),(2,50),(3,20),(3,35),(3,50)])


def corners_to_range_descriptions(*, start_corner=None, stop_corner=None, step_corner=None, automatic_step_sign=False):

    assert stop_corner is not None
    if start_corner is None:
        start_corner = tuple(0 for i in range(len(stop_corner)))
    if step_corner is None:
        # step_corner = tuple(1 for i in range(len(stop_corner)))
        srcs = (start_corner, stop_corner)
    else:
        srcs = (start_corner, stop_corner, step_corner)
    descriptions = list(izip_uniform_containers(*srcs))
    
    for i, description in enumerate(descriptions):
        if description[1] < description[0]:
            if len(description) == 2:
                if automatic_step_sign:
                    descriptions[i] = description + (-1,)
                else:
                    raise ValueError("for axis {}, start > stop, and step sign is missing, but automatic step signs are disabled.".format(i))
            else:
                assert len(description) == 3
                if not description[2] < 0:
                    raise ValueError("for axis {}, start > stop, but step is not negative. description={}.".format(repr(description)))
                    
    return descriptions


def higher_range_by_corners(*, iteration_order=None, **other_kwargs):
    descriptions = corners_to_range_descriptions(**other_kwargs)
    return higher_range(descriptions, iteration_order=iteration_order)
    
assert_equal(list(higher_range_by_corners(start_corner=(5,50), stop_corner=(7,52))), [(5,50), (5,51), (6,50), (6,51)])
assure_raises_instanceof(higher_range_by_corners, ValueError)(start_corner=(5,52), stop_corner=(7,50))
assert_equal(list(higher_range_by_corners(start_corner=(5,52), stop_corner=(7,50), automatic_step_sign=True)), [(5,52), (5,51), (6,52), (6,51)])
assert_equal(list(higher_range_by_corners(start_corner=(5,50), stop_corner=(7,52), iteration_order=(0,1))), [(5,50), (6,50), (5,51), (6,51)])

