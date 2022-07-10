import itertools

import collections

from TestingAtoms import assert_equal, AssuranceError, AlternativeAssertionError, summon_cactus
from TestingBasics import assure_raises_instanceof




class ProvisionError(Exception):
    pass

class MysteriousError(Exception):
    """ don't catch this. Just identify its cause and replace it with a better exception. and then maybe catch it. """
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
                if inputGenIndex == 0:
                    for genIndexB, genB in enumerate(inputGens):
                        try:
                            assert_empty(genB)
                        except AssertionError:
                            raise AssuranceError(f"the generators did not run out of items all at the same time, at item index {itemIndex}.") from None
                    return # they all ran out at the same time.
                else:
                    raise AssuranceError(f"generator at index {inputGenIndex} had no item at index {itemIndex}!") 
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
    
    





