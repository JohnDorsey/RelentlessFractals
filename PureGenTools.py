
def _assert_equal(thing0, thing1): # a copy.
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)

import collections


def peek_first_and_iter(input_seq):
    inputGen = iter(input_seq)
    try:
        first = next(inputGen)
    except StopIteration:
        raise IndexError("empty, couldn't peek first!")
    return (first, inputGen)


    




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
        
        
def gen_track_previous_full(input_seq):
    try:
        previousItem, inputGen = peek_first_and_iter(input_seq)
    except IndexError:
        raise IndexError("can't fill! not enough items!") # this used to return None without error.
    for currentItem in inputGen:
        yield (previousItem, currentItem)
        previousItem = currentItem
assert (list(gen_track_previous_full(range(5,10))) == [(5,6), (6,7), (7,8), (8,9)])


def gen_track_recent(input_seq, count=None):
    history = collections.deque([None for i in range(count)])
    for i, item in enumerate(input_seq):
        history.append(item)
        history.popleft()
        yield tuple(history)
assert list(gen_track_recent("abcdef", count=3)) == [(None, None, "a"), (None, "a", "b"), ("a","b","c"), ("b","c","d"),("c","d","e"),("d","e","f")]
        

def gen_track_recent_full(input_seq, count=None):
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
    waste = tuple(None for i in range(count))
    while waste.count(None) > 1:
        try:
            waste = next(result)
        except StopIteration:
            raise IndexError("could not!?")
    assert waste.count(None) == 1
    assert waste[0] is None
    return result
assert (list(gen_track_recent_full("abcdef", count=3)) == [("a","b","c"),("b","c","d"),("c","d","e"),("d","e","f")])
    
    
    
    
                

        
"""
def gen_track_previous_tuple_flatly(input_seq):
    previousTuple = None
    for item in input_seq:
        if not isinstance(item, tuple):
            item = (item,)
        if previousTuple is None:
            yield (None,)*len(item) + item
        else:
            yield previousTuple + item
        previousTuple = item
        assert isinstance(previousTuple, tuple)
        
        
def enumerate_flatly(input_seq, start=0):
    for i, item in enumerate(input_seq, start=start):
        if isinstance(item, tuple):
            yield (i,) + item
        else:
            yield (i, item)
"""