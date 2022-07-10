


class AlternativeAssertionError(Exception):
    """ this is used while testing methods that are supposed to raise assertion errors. """
    pass
    
class AlternativeError(Exception):
    """ this is raised by methods that are supposed to raise exceptions, when something else goes wrong. """
    pass
    

class AssuranceError(Exception):
    pass

"""
class ImplicitAssuranceError(Exception):
    # should this be used in place of assurance errors when the function raising the assurance error doesn't contain "assure" in the name? maybe not, as it could allow more errors when renaming.
    pass
"""
"""
def assert_dicts_are_equal(*things, start_message="", end_message=""):
    if len(things) != 2:
        raise AlternativeAssertionError("too few items")
    
    baseMessage = ""
    if set(things[0].keys()) != set(things[0].keys():
        baseMessage += " Their key sets differ.{}{}.".format(" left set extras: {}.".format() if
"""
def assert_equal(*things, start_message="", message=""):
    if len(things) < 2:
        raise AlternativeAssertionError("too few items.")
    elif len(things) == 2:
        if things[0] != things[1]:
            baseMessage = "{} does not equal {}.".format(things[0], things[1])
            if type(things[0]) != type(things[0]):
                baseMessage += " Their types differ ({} and {}).".format(type(things[0]), type(things[1]))
            if hasattr(things[0], "__len__") and hasattr(things[1], "__len__"):
                if len(things[0]) != len(things[1]):
                    baseMessage += " Their lengths differ ({} and {}).".format(len(things[0]), len(things[1]))
                else:
                    baseMessage += " Their lengths are the same."
            if all(isinstance(thing, dict) for thing in things):
                baseMessage += " Both are dicts."
                keySets = tuple(thing.keys() for thing in things)
                if keySets[0] != keySets[1]:
                    keySetExtras = [keySets[i].difference(keySets[1-i]) for i in (0,1)]
                    baseMessage += " Their key sets differ.{}{}.".format(
                            " left set extras: {}.".format(keySetExtras[0]) if len(keySetExtras[0]) else "",
                            " right set extras: {}.".format(keySetExtras[1]) if len(keySetExtras[1]) else "",
                        )
                else:
                    baseMessage += " Their key sets are the same."
                    for key in things[0].keys():
                        assert_equal(things[0][key], things[1][key], start_message=start_message+" "+baseMessage+(" the values for key {} are unequal:\n".format(key)), message=message)
                    raise AlternativeAssertionError("Two dicts were not equal, but all of their items were equal?")
            raise AssertionError(start_message + baseMessage + message)
    else:
        for i in range(len(things)-1):
            assert_equal(things[i], things[i+1], start_message=start_message, message=" (at comparison {} in chain).".format(i)+message)
    
    

def assert_less(thing0, thing1, message=""):
    assert thing0 < thing1, "{} is not less than {}.".format(thing0, thing1)+message

def assert_isinstance(thing0, reference_class, message=""):
    assert isinstance(thing0, reference_class), "{} of type {} is not an instance of {}.".format(repr(thing0), repr(type(thing0)), repr(reference_class))+message

def assure_isinstance(thing0, reference_class, message=""):
    try:
        assert_isinstance(thing0, reference_class, message=message)
    except AssertionError as ae:
        raise AssuranceError(ae)
    return thing0
    
def gen_assure_areinstances(input_seq, reference_class, message=""):
    for i, item in enumerate(input_seq):
        if not isinstance(item, reference_class):
            raise AssertionError(f"{repr(item)} of type {type(item)} at index {i} is not an instance of {reference_class}."+message)
        yield item
        
assert list(gen_assure_areinstances(range(5), int)) == [0,1,2,3,4]
# assert_raises_instanceof(wrap_with(gen_assure_areinstances, list), AssertionError)(range(5), (str,float))




    

def summon_cactus(message, _persistent_cacti=dict()):
    """
    Make the origin of a value representing an error (such as a list item placeholder that should always be overwritten before returning the list) easier to track down by creating such values on the fly with a very descriptive type name that will appear in any TypeError they cause. Because they have no attributes, they cause such errors earlier and leave shorter stack traces.
    go from
        "TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'"
    to
        "TypeError: unsupported operand type(s) for +: 'int' and 'placeholder_for_ReorderComplexListByKey'"
        _or_ a similar error which occurs earlier, like an attribute error for __str__ which would not have been caused by str(None).
        
    persistent cacti help avoid overhead from re-creating the same cactus over and over.
        As a side effect,
            cactusA=summon_cactus("specific text");
            cactusB=summon_cactus("specific_text");
            (cacusA is cactusB) --> True
       but this is not a standard behavior and might change, and can also be broken by reloading the module. So probably don't rely on it at all.
    
    putting spaces in the name of a type doesn't seem to cause any errors, but it's weird and distracting, so I choose not to.
    """
    if message in _persistent_cacti:
        return _persistent_cacti[message]
    else:
        _persistent_cacti[message] = type(message, (), dict())()
        if len(_persistent_cacti) >= 64 and len(_persistent_cacti) in [2**n for n in range(6,24)]:
            print("summon_cactus: warning: {} unique cacti sure is a lot. Is memory being wasted by their mass-production? If not, adjust this warning threshold.".format(len(_persistent_cacti)))
        return _persistent_cacti[message]
        
        
def raise_inline(*args):
    if len(args) == 1:
        errorToRaise = args[0]
    elif len(args) == 2:
        errorToRaise = args[0](args[1])
    else:
        raise AlternativeError(f"wrong number of args for raise_inline: expected 1 or 2, got {len(args)}.")
    
    if not isinstance(errorToRaise, Exception):
        raise AlternativeError(f"bad argument type with {len(args)} arg(s).")
    if isinstance(errorToRaise, (AlternativeError, AlternativeAssertionError)):
        print("raise_inline: warning: AlternativeError or AlternativeAssertionError should NEVER be given to raise_inline as inputs, because they are also raised when there are problems with the input arguments. They will be raised anyway.")
    raise errorToRaise
    
    



