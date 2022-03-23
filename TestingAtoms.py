


class AlternativeAssertionError(Exception):
    # this is used while testing methods that are supposed to raise assertion errors.
    pass
    

class AssuranceError(Exception):
    pass
    


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