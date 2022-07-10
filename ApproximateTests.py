




COMPLEX_ERROR_TOLERANCE = (2**-36)




def test_complex_nearly_equal(val0, val1, error_tolerance=COMPLEX_ERROR_TOLERANCE, debug=False):
    err = val0 - val1
    errMagnitude = abs(err)
    if errMagnitude == 0:
        return True
    elif errMagnitude < error_tolerance:
        # print("warning: {} and {} are supposed to be equal.".format(val0, val1))
        return True
    else:
        if debug:
            print("test_complex_nearly_equal: debug: {} is not nearly equal to {}, err is {}, errMagnitude is {}.".format(val0, val1, err, errMagnitude))
        return False
        

def _assert_complex_nearly_equal(val0, val1, error_tolerance=COMPLEX_ERROR_TOLERANCE):
    assert test_complex_nearly_equal(val0, val1, error_tolerance=error_tolerance, debug=True), "{} is not close enough to {} with tolerance setting {}.".format(val0, val1, error_tolerance)
    """
    "the difference between {} and {} is {} with length {} - that's {} times greater than the error tolerance {}.".format(
            val0, val1, err, errMagnitude, str(errMagnitude/error_tolerance),
        )
    """


def test_nearly_equal(thing0, thing1, error_tolerance=COMPLEX_ERROR_TOLERANCE, debug=False):
    head = "test_nearly_equal: debug: "
    if isinstance(thing0, (complex,float,int)) and isinstance(thing1, (complex,float,int)):
        result = test_complex_nearly_equal(thing0, thing1, error_tolerance=error_tolerance)
        if debug and not result:
            print(head + "failed in br0.")
        return result
    elif any(isinstance(thing0, testEnterable) and isinstance(thing1, testEnterable) for testEnterable in (tuple, list)):
        if len(thing0) != len(thing1):
            if debug:
                print(head + "lengths differ.")
            return False
        result = all(test_nearly_equal(thing0[i], thing1[i], error_tolerance=error_tolerance, debug=debug) for i in range(max(len(thing0), len(thing1))))
        if debug and not result:
            print(head + "failed in br1.")
        return result
    else:
        result = (thing0 == thing1)
        if debug and not result:
            print(head + "failed in br2.")
        return result


def assert_nearly_equal(thing0, thing1, error_tolerance=COMPLEX_ERROR_TOLERANCE):
    assert test_nearly_equal(thing0, thing1, error_tolerance=error_tolerance, debug=True), "{} does not nearly equal {} with error tolerance {}.".format(repr(thing0), repr(thing1), repr(error_tolerance))







