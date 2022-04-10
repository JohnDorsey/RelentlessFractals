
import operator


from PureGenTools import peek_first_and_iter, ProvisionError, gen_track_previous






def all_are_equal_to(input_seq, *, example=None, equality_test_fun=operator.eq):
    for item in input_seq:
        if not equality_test_fun(example, item):
            return False
    return True
assert all_are_equal_to([2,3], example=2) == False
assert all_are_equal_to([], example=2) == True
assert all_are_equal_to([1], example=2) == False


def all_are_equal(input_seq, equality_test_fun=operator.eq):
    first, inputGen = peek_first_and_iter(input_seq)
    return all_are_equal_to(inputGen, example=first, equality_test_fun=equality_test_fun)

"""
erb = get_exception_raised_by(all_are_equal)([])
assert isinstance(erb, ProvisionError), repr(erb)
assert all_are_equal("aaaaa")
assert not all_are_equal("aaaba")
del erb
"""

    
def get_shared_value(input_seq, equality_test_fun=operator.eq):
    result, inputGen = peek_first_and_iter(input_seq)
    for i, item in enumerate(inputGen):
        if not equality_test_fun(result, item):
            raise AssuranceError("at index {}, item value {} does not equal shared value {}.".format(i, repr(item), repr(result)))
    return result

assert get_shared_value("aaaaa") == "a"
"""
assert_raises_instanceof(get_shared_value, AssuranceError)("aaaba")
"""



def ints_are_consecutive_increasing(input_seq):
    """
    first, inputGen = peek_first_and_iter(input_seq)
    if first
    for compNum, newNum in enumerate(inputGen, start=first+1):
        if new
        if not compNum == newNum:
    """
    for i, (previous, current) in enumerate(gen_track_previous(input_seq)):
        assert isinstance(current, int)
        if i == 0:
            continue
        else:
            if current != previous + 1:
                return False
    return True

assert ints_are_consecutive_increasing([5,6,7,8,9])
assert not ints_are_consecutive_increasing([5,6,7,8,10])
assert not ints_are_consecutive_increasing([9,8,7,6,5])

    
def ints_are_contiguous(input_seq):
    return ints_are_consecutive_increasing(sorted(input_seq))
            
assert ints_are_contiguous([5,8,6,7])
assert ints_are_contiguous([-2,1,-1,0,-3])
assert not ints_are_contiguous([3,4,2,0,-1])
