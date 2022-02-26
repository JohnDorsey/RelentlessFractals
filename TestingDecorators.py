
import copy
import itertools
import operator

from TestingBasics import get_shared_value

from ComplexGeometry import multi_traverse




def fuzz_inputs_share_output(input_fun, fuzzer_gen, equality_test_fun=None):
    """
    def inner(*args, **kwargs):
        testResultGen = (input_fun(*fuzzer(args), **kwargs) for fuzzer in fuzzer_gen)
        return get_shared_value(testResultGen)
    return inner
    """
    defuzzerGen = itertools.repeat((lambda x: x))
    return fuzz_inputs_share_defuzzed_output(input_fun, zip(fuzzer_gen, defuzzerGen), equality_test_fun=equality_test_fun)


def fuzz_inputs_share_defuzzed_output(input_fun, fuzzer_defuzzer_pair_gen, equality_test_fun=None):
    def inner(*args, **kwargs):
        testResultGen = (inverse_fuzzer(input_fun(*fuzzer(args), **kwargs)) for fuzzer, inverse_fuzzer in fuzzer_defuzzer_pair_gen)
        return get_shared_value(testResultGen, equality_test_fun=equality_test_fun)
    return inner
        




def all_products_from_seq_pair(data0, data1):
    for itemA in data0:
        for itemB in data1:
            yield itemA*itemB
assert list(all_products_from_seq_pair([1,2],[3,5,100])) == [3, 5, 100, 6, 10, 200]


def complex_parallel_product(values):
    result = complex(1,1)
    for value in values:
        result = complex(result.real*value.real, result.imag*value.imag)
    return result
assert complex_parallel_product([1+2j,5+100j]) == 5+200j


def complex_pair_parallel_div(val0, val1):
    return complex(val0.real/val1.real, val0.imag/val1.imag)
    

def make_transformed_copy(data, enter_trigger_fun=None, transform_trigger_fun=None, transform_fun=None):
    if enter_trigger_fun is None:
        def enter_trigger_fun(testItem):
            if isinstance(testItem, (tuple,list)):
                assert not transform_trigger_fun(testItem)
                return True
            else:
                return False
    if enter_trigger_fun(data):
        return type(data)(make_transformed_copy(item, enter_trigger_fun=enter_trigger_fun, transform_trigger_fun=transform_trigger_fun, transform_fun=transform_fun) for item in data)
    elif transform_trigger_fun(data):
        return transform_fun(data)
    else:
        return copy.deepcopy(data)
    



def gen_basic_complex_fuzzers_and_inverses(include_neutral=True):
    neutralCounter = 0
    for rOff, iOff in multi_traverse((-1.0, 0.0, 1.0), count=2):
        for rScale, iScale in multi_traverse(list(all_products_from_seq_pair((0.5, 1.0, 2.0), (-1.0, 1.0))), count=2):
            for complexScale in (1.0+0.0j, 0.0+0.7j):
                isNeutral = ((rOff, iOff, rScale, iScale, complexScale) == (0.0, 0.0, 1.0, 1.0, 1.0+0.0j))
                if isNeutral:
                    neutralCounter += 1
                    if not include_neutral:
                        continue
                
                def currentFun(inputArgs):
                    return make_transformed_copy(
                        inputArgs,
                        transform_trigger_fun=(lambda x: isinstance(x, complex)),
                        transform_fun=(lambda w: complex_parallel_product([w+complex(rOff, iOff), complex(rScale, iScale)])*complexScale),
                    )
                def currentInverseFun(inputArgs):
                    return make_transformed_copy(
                        inputArgs,
                        transform_trigger_fun=(lambda x: isinstance(x, complex)),
                        transform_fun=(lambda w: complex_pair_parallel_div(w/complexScale, complex(rScale, iScale))-complex(rOff, iOff)),
                    )
                    
                yield (currentFun, currentInverseFun)
    assert neutralCounter == 1
                
for testFun, testInverseFun in gen_basic_complex_fuzzers_and_inverses():
    assert testInverseFun(testFun(complex(10,10000))) == complex(10,10000)
    

def basic_complex_fuzz_inputs_only(input_fun, equality_test_fun=None):
    """
    def fuzzedArgTupleGenFun(inputArgs):
        for fuzzFun, _ in gen_basic_complex_fuzzers_and_inverses():
            yield fuzzFun(inputArgs)
    """
    fuzzerGen = (pair[0] for pair in gen_basic_complex_fuzzers_and_inverses())
    inner = fuzz_inputs_share_output(input_fun, fuzzerGen, equality_test_fun=equality_test_fun)
    return inner

testList = []
def testAppender(inputItem):
    testList.append(inputItem)
basic_complex_fuzz_inputs_only(testAppender, equality_test_fun=operator.eq)(100.0+10000.0j)
assert len(set(testList)) == 9*(3*2*3*2)*2
testList.clear()
basic_complex_fuzz_inputs_only(testAppender, equality_test_fun=operator.eq)([["a", (200.0+20000.0j,)], (5, 6), "b"])
for item in testList:
    assert item[-2:] == [(5,6), "b"]
assert len(set(str(item) for item in testList)) == 9*3*2*3*2*2
del testList
del testAppender


def basic_complex_fuzz_io(input_fun, equality_test_fun=operator.eq):
    inner = fuzz_inputs_share_defuzzed_output(input_fun, gen_basic_complex_fuzzers_and_inverses(), equality_test_fun=equality_test_fun)
    return inner



