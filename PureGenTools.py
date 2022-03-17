import itertools


def _assert_equal(thing0, thing1): # a copy to prevent circular imports.
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)

import collections



class ProvisionError(Exception):
    pass

class MysteriousError(Exception):
    # don't catch this. Just identify its cause and replace it with a better exception. and then maybe catch it.
    pass


def peek_first_and_iter(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        raise ProvisionError()
    return (first, inputGen)


def assert_empty(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        return
    assert False, "input seq was not empty, first item was {}.".format(repr(first))






testZip = zip("ab","cd")
izip_shortest = (zip if (iter(testZip) is iter(testZip)) else itertools.izip)
testZip2 = izip_shortest("ab","cd")
assert (iter(testZip2) is iter(testZip2)) and (not isinstance(testZip2, list)), "can't izip?"
del testZip, testZip2

try:
    izip_longest = itertools.izip_longest
except AttributeError:
    izip_longest = itertools.zip_longest

    




def higher_range_linear(descriptions):
            
    assert len(descriptions) > 0
    
    if len(descriptions) == 1:
        for i in range(*descriptions[0]):
            yield (i,)
    else:
        for i in range(*descriptions[0]):
            for extension in higher_range_linear(descriptions[1:]):
                yield (i,) + extension
assert list(higher_range_linear([(2,5), (3,10,3)])) == [(a,b) for a in range(2,5) for b in range(3,10,3)] == [(2,3), (2,6), (2,9), (3,3), (3,6), (3,9), (4,3), (4,6), (4,9)]


def higher_range(descriptions, iteration_order=None):
    if iteration_order is not None:
        assert len(iteration_order) == len(descriptions)
        assert sorted(iteration_order) == list(range(len(iteration_order)))
        reorderedDescriptions = [None for i in range(len(descriptions))]
        for srcIndex, destIndex in enumerate(iteration_order):
            reorderedDescriptions[destIndex] = descriptions[srcIndex]
        for unorderedItem in higher_range_linear(reorderedDescriptions):
            reorderedItem = tuple(unorderedItem[srcIndex] for srcIndex in iteration_order)
            yield reorderedItem
    else:
        for item in higher_range_linear(descriptions):
            yield item
            
_assert_equal(list(higher_range_linear([(2,5), (3,10,3), (4,)])), [(a,b,c) for a in range(2,5) for b in range(3,10,3) for c in range(4)])
assert list(higher_range([(2,5), (3,10,3), (4,)], iteration_order=[2,0,1])) == [(a,b,c) for b in range(3,10,3) for c in range(4) for a in range(2,5)]


            
            
def gen_track_previous(input_seq):
    previousItem = None
    for item in input_seq:
        yield (previousItem, item)
        previousItem = item
assert (list(gen_track_previous(range(5,10))) == [(None,5),(5,6),(6,7),(7,8),(8,9)])
        
        
def gen_track_previous_full(input_seq, allow_waste=False):
    try:
        previousItem, inputGen = peek_first_and_iter(input_seq)
    except ProvisionError:
        if allow_waste:
            return
        else:
            raise MysteriousError("can't fill! not enough items!")
    for currentItem in inputGen:
        yield (previousItem, currentItem)
        previousItem = currentItem
assert (list(gen_track_previous_full(range(5,10))) == [(5,6), (6,7), (7,8), (8,9)])


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
_assert_equal(list(gen_track_recent_trimmed("abcdef", count=3)), [("a",), ("a", "b"), ("a","b","c"), ("b","c","d"),("c","d","e"),("d","e","f")])


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
                raise MysteriousError("could not do the thing!")
    assert trash.count(None) == 1
    assert trash[0] is None
    return result
assert (list(gen_track_recent_full("abcdef", count=3)) == [("a","b","c"),("b","c","d"),("c","d","e"),("d","e","f")])
assert (list(gen_track_recent_full("abc", count=5, allow_waste=True)) == [])
    
    
    
    
    
    
    
    
    
    
    
    
    
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
_assert_equal(list(enumerate_to_depth_unpacked([5,6,7,8], depth=1)), [(0,5), (1,6), (2,7), (3,8)])
_assert_equal(list(enumerate_to_depth_unpacked([[5,6],[7,8]], depth=2)), [(0,0,5), (0,1,6), (1,0,7), (1,1,8)])



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
_assert_equal(list(enumerate_to_depth_packed([5,6,7,8], depth=1)), [((0,),5), ((1,),6), ((2,),7), ((3,),8)])
_assert_equal(list(enumerate_to_depth_packed([[5,6],[7,8]], depth=2)), [((0,0),5), ((0,1),6), ((1,0),7), ((1,1),8)])



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
_assert_equal(list(iterate_to_depth([[2,3], [4,5], [[6,7], 8, [9,10]]], depth=2)), [2,3,4,5,[6,7],8,[9,10]])










def gen_chunks_as_lists(data, length):
    itemGen = iter(data)
    while True:
        chunk = [item for item in itertools.islice(itemGen, 0, length)]
        if len(chunk) == 0:
            return
        yield chunk
        if len(chunk) < length:
            assert_empty(itemGen)
            return
        else:
            assert len(chunk) == length
    assert False
    
assert list(gen_chunks_as_lists(range(9), 2)) == [[0,1], [2,3], [4,5], [6,7], [8]]
assert list(gen_chunks_as_lists(range(8), 2)) == [[0,1], [2,3], [4,5], [6,7]]


