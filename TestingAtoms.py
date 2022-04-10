


class AlternativeAssertionError(Exception):
    # this is used while testing methods that are supposed to raise assertion errors.
    pass
    

class AssuranceError(Exception):
    pass

"""
class ImplicitAssuranceError(Exception):
    # should this be used in place of assurance errors when the function raising the assurance error doesn't contain "assure" in the name? maybe not, as it could allow more errors when renaming.
    pass
"""
    


def assert_equal(*things, message=""):
    if len(things) < 2:
        raise AlternativeAssertionError("too few items.")
    elif len(things) == 2:
        assert things[0] == things[1], "{} does not equal {}.".format(things[0], things[1]) + message
    else:
        for i in range(len(things)-1):
            assert_equal(things[i], things[i+1], message=" (at comparison {} in chain).".format(i)+message)
    
def assert_less(thing0, thing1, message=""):
    assert thing0 < thing1, "{} is not less than {}.".format(thing0, thing1)+message

def assert_isinstance(thing0, reference_class, message=""):
    assert isinstance(thing0, reference_class), "{} of type {} is not an instance of {}.".format(repr(thing0), repr(type(thing0)), repr(reference_class))+message

def assure_isinstance(thing0, reference_class, message=""):
    try:
        assert_isinstance(thing0, reference_class, message=message)
    except AssertionError as ae:
        raise AssuranceError(ae.message)
    return thing0
    
    
    

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
        
        




