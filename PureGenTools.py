import itertools

import collections

from TestingAtoms import assert_equal, AssuranceError, AlternativeAssertionError, summon_cactus
from TestingBasics import assure_raises_instanceof




class ProvisionError(Exception):
    pass

class MysteriousError(Exception):
    # don't catch this. Just identify its cause and replace it with a better exception. and then maybe catch it.
    pass


def take_first_and_iter(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        raise ProvisionError()
    return (first, inputGen)

assure_raises_instanceof(take_first_and_iter, ProvisionError)([])
assert take_first_and_iter(range(2, 10))[0] == 2


def assert_empty(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        return
    assert False, "input seq was not empty, first item was {}.".format(repr(first))

assert_empty((item for item in []))
"""
try:
    assert_empty([5])
    raise AlternativeAssertionError() # just because it is never caught, but this isn't its purpose.
except AssertionError:
    pass
"""
assure_raises_instanceof(assert_empty, AssertionError)([5])


def wrap_with(input_fun, wrapper):
    """ this is helpful in testing whether a generator eventually raises an error. """
    def wrap_with_inner(*args, **kwargs):
        return wrapper(input_fun(*args, **kwargs))
    return wrap_with_inner
    
assert_equal(wrap_with(sum, (lambda x: x**2))([1,2,3]), 36)


testZip = zip("ab","cd")
izip_shortest = (zip if (iter(testZip) is iter(testZip)) else itertools.izip)
testZip2 = izip_shortest("ab","cd")
assert (iter(testZip2) is iter(testZip2)) and (not isinstance(testZip2, list)), "can't izip?"
del testZip, testZip2




try:
    izip_longest = itertools.izip_longest
except AttributeError:
    izip_longest = itertools.zip_longest

"""
def izip_uniform(*input_seqs):
    raise NotImplementedError("doesn't work!")
    inputGens = [iter(inputSeq) for inputSeq in input_seqs]
    outputGen = izip_shortest(*inputGens)
    for item in outputGen:
        yield item
    
    failData = set()
    for i,inputGen in enumerate(inputGens):
        try:
            assert_empty(inputGen)
        except AssertionError:
            failData.add(i)
    if len(failData)> 0:
        raise AssuranceError("The following seq(s) were not empty: {}.".format(failData))
"""
"""
def get_next_of_each(input_gens):
    try:
        return tuple(next(inputGen) for inputGen in inputGens)
    except StopIteration:
        raise 
"""


def izip_uniform(*input_seqs):
    inputGens = list(map(iter, input_seqs))
    currentBucket = []
    for itemIndex in itertools.count():
        currentBucket.clear()
        for inputGenIndex, inputGen in enumerate(inputGens):
            try:
                currentBucket.append(next(inputGen))
            except StopIteration:
                if inputGenIndex != 0:
                    raise AssuranceError(f"generator at index {inputGenIndex} had no item at index {itemIndex}!")
                else:
                    for genIndexB, genB in enumerate(inputGens):
                        try:
                            assert_empty(genB)
                        except AssertionError:
                            raise AssuranceError(f"the generators did not run out of items all at the same time, at item index {itemIndex}.")
                    # they all ran out at the same time.
                    return
            # continue to next gen.
        yield tuple(currentBucket)
    assert False

assert_equal(list(izip_uniform("abcdefg", [1,2,3,4,5,6,7])), list(zip("abcdefg", [1,2,3,4,5,6,7])))
assure_raises_instanceof(wrap_with(izip_uniform, list), AssuranceError)("abcdefg", [1,2,3,4,5,6])


def izip_uniform_containers(*input_containers):
    sharedLength = len(input_containers[0])
    assert all(hasattr(item, "__len__") for item in input_containers), f"these containers don't all have __len__. their types are {[type(item) for item in input_containers]}."
    assert all(len(other)==sharedLength for other in input_containers[1:]), "the items don't all have the same lengths."
    return izip_shortest(*input_containers)



def apply_slice_chain(data, slice_chain):
    for currentSlice in slice_chain:
        data = data[currentSlice]
    return data

def higher_range_linear(descriptions, *, post_slices=None):
    # this might be a reinvention of itertools.product.
    assert len(descriptions) > 0
    ranges = [range(*description) for description in descriptions]
    if post_slices is not None:
        for i, (currentRange, currentSlices) in enumerate(izip_uniform(ranges, post_slices)):
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
    if not reverse_output:
        raise NotImplementedError("not tested with reverse_output=False")
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
    descriptions = list(izip_uniform(*srcs))
    
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





            
def gen_track_previous(input_seq):
    previousItem = None
    for item in input_seq:
        yield (previousItem, item)
        previousItem = item
        
assert (list(gen_track_previous(range(5,10))) == [(None,5),(5,6),(6,7),(7,8),(8,9)])
        
        
def gen_track_previous_full(input_seq, allow_waste=False):
    try:
        previousItem, inputGen = take_first_and_iter(input_seq)
    except ProvisionError:
        raise MysteriousError("can't fill, because there are no items.")
    try:
        currentItem = next(inputGen)
    except StopIteration:
        if allow_waste:
            return
        else:
            raise MysteriousError("waste would happen, but is not allowed.")
    yield (previousItem, currentItem)
    previousItem = currentItem
    for currentItem in inputGen:
        yield (previousItem, currentItem)
        previousItem = currentItem
        
assert_equal(list(gen_track_previous_full(range(5,10))), [(5,6), (6,7), (7,8), (8,9)])
assure_raises_instanceof(wrap_with(gen_track_previous_full, list), MysteriousError)([5])
assure_raises_instanceof(wrap_with(gen_track_previous_full, list), MysteriousError)([])



def gen_track_recent(input_seq, count=None, default=None):
    history = collections.deque([default for i in range(count)])
    for item in input_seq:
        history.append(item)
        history.popleft()
        yield tuple(history)
        
assert list(gen_track_recent("abcdef", count=3, default=999)) == [(999, 999, "a"), (999, "a", "b"), ("a","b","c"), ("b","c","d"),("c","d","e"),("d","e","f")]


def gen_track_recent_trimmed(input_seq, count=None):
    history = collections.deque([])
    for item in input_seq:
        history.append(item)
        while len(history) > count:
            history.popleft()
        yield tuple(history)
        
assert_equal(list(gen_track_recent_trimmed("abcdef", count=3)), [("a",), ("a", "b"), ("a","b","c"), ("b","c","d"),("c","d","e"),("d","e","f")])


def gen_track_recent_full(input_seq, count=None, allow_waste=False):
    assert count >= 2
    result = gen_track_recent(input_seq, count=count)
    """
    i, Waste = (None, None)
    for i in range(count-1,0,-1):
        try:
            waste = next(result)
            assert waste.count(None) == i
        except StopIteration:
            raise IndexError("could not!?")
    assert i == 1
    """
    trash = tuple(None for i in range(count))
    while trash.count(None) > 1:
        try:
            trash = next(result)
        except StopIteration:
            if allow_waste:
                return ()
            else:
                raise MysteriousError(f"not enough items to yield a full batch of {count} items.")
    assert trash.count(None) == 1
    assert trash[0] is None
    return result
    
assert (list(gen_track_recent_full("abcdef", count=3)) == [("a","b","c"),("b","c","d"),("c","d","e"),("d","e","f")])
assert (list(gen_track_recent_full("abc", count=5, allow_waste=True)) == [])
assure_raises_instanceof(wrap_with(gen_track_recent_full, list), MysteriousError)("abc", count=5, allow_waste=False)
    
    
    
    
    
    
    
    
    
    
    
    
def enumerate_to_depth_unpacked(data, depth=None):
    assert depth > 0
    if depth == 1:
        for pair in enumerate(data): # return can't be used because yield appears in other branch. This does NOT produce error messages in python 3.8.10.
            yield pair
    else:
        assert depth > 1
        for i, item in enumerate(data):
            for longItem in enumerate_to_depth_unpacked(item, depth=depth-1):
                yield (i,) + longItem
                
assert_equal(list(enumerate_to_depth_unpacked([5,6,7,8], depth=1)), [(0,5), (1,6), (2,7), (3,8)])
assert_equal(list(enumerate_to_depth_unpacked([[5,6],[7,8]], depth=2)), [(0,0,5), (0,1,6), (1,0,7), (1,1,8)])



def enumerate_to_depth_packed(data, depth=None):
    assert depth > 0
    if depth == 1:
        for i, item in enumerate(data):
            yield ((i,), item)
    else:
        assert depth > 1
        for i, item in enumerate(data):
            for subItemAddress, subItem, in enumerate_to_depth_packed(item, depth=depth-1):
                yield ((i,)+subItemAddress, subItem)
                
assert_equal(list(enumerate_to_depth_packed([5,6,7,8], depth=1)), [((0,),5), ((1,),6), ((2,),7), ((3,),8)])
assert_equal(list(enumerate_to_depth_packed([[5,6],[7,8]], depth=2)), [((0,0),5), ((0,1),6), ((1,0),7), ((1,1),8)])



def iterate_to_depth(data, depth=None):
    assert depth > 0
    if depth == 1:
        for item in data: # return can't be used because yield appears in other branch. This does NOT produce error messages in python 3.8.10.
            yield item
    else:
        assert depth > 1
        for item in data:
            for subItem in iterate_to_depth(item, depth=depth-1):
                yield subItem
                
assert_equal(list(iterate_to_depth([[2,3], [4,5], [[6,7], 8, [9,10]]], depth=2)), [2,3,4,5,[6,7],8,[9,10]])










def gen_chunks_as_lists(data, length, allow_partial=True):
    itemGen = iter(data)
    while True:
        chunk = list(itertools.islice(itemGen, 0, length))
        if len(chunk) == 0:
            assert_empty(itemGen)
            return
        elif len(chunk) == length:
            yield chunk
        else:
            assert 0 < len(chunk) < length
            assert_empty(itemGen)
            if not allow_partial:
                raise AssuranceError("the last chunk was partial. it contained {} of the required {} items.".format(len(chunk), length))
            yield chunk
            return
    assert False
    
assert list(gen_chunks_as_lists(range(9), 2)) == [[0,1], [2,3], [4,5], [6,7], [8]]
assert list(gen_chunks_as_lists(range(8), 2)) == [[0,1], [2,3], [4,5], [6,7]]
assure_raises_instanceof(wrap_with(gen_chunks_as_lists, list), AssuranceError)(range(9), 2, allow_partial=False)



def get_next_assuredly_available(input_gen):
    try:
        result = next(input_gen)
    except StopIteration:
        raise AssuranceError("no next item was available!") from None
    return result
        
assert_equal(get_next_assuredly_available(iter(range(2,5))), 2)
assure_raises_instanceof(get_next_assuredly_available, AssuranceError)(iter(range(0)))
        

def get_next_assuredly_last(input_gen):
    result = get_next_assuredly_available(input_gen)
    try:
        assert_empty(input_gen)
    except AssertionError as ate:
        raise AssuranceError(f"more items remained. assert_empty says: {ate}") from None
    return result

assure_raises_instanceof(get_next_assuredly_last, AssuranceError)(iter(range(5)))
assure_raises_instanceof(get_next_assuredly_last, AssuranceError)(iter(range(0)))
    

def yield_next_assuredly_last(input_gen):
    yield get_next_assuredly_last(input_gen)


def assure_gen_length_is(input_gen, length):
    assert length > 0
    assert iter(input_gen) is iter(input_gen)
    return itertools.chain(itertools.slice(input_gen, length-1), yield_next_assuredly_last(input_gen))
    
    
"""
def yield_next_assuredly_exists(input_gen):
    try:
        result = next(input_gen)
    except StopIteration:
        raise AssuranceError("next item did not exist!")
    yield result
"""

def gen_assure_never_exhausted(input_seq):
    i = -1
    for i, item in enumerate(input_seq):
        yield item
    raise AssuranceError("input_seq was exhausted after {} items.".format(i+1))
    
    
def islice_assuredly_full(input_seq, *other_args, **other_kwargs):
    """
    assert length >= 1
    for i, item in enumerate(input_seq):
        yield item
        if i+1 == length:
            return
    raise AssuranceError("a full slice could not be made.")
    """
    return itertools.islice(gen_assure_never_exhausted(input_seq), *other_args, **other_kwargs)
    
    





