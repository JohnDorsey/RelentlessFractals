KEYBOARD_DIGITS = "0123456789"
KEYBOARD_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
KEYBOARD_SYMBOLS = " `~!@#$%^&*()-_=+[{]}\\|\\;:'\",<.>/?"
KEYBOARD_CHARS = KEYBOARD_DIGITS + KEYBOARD_LETTERS + KEYBOARD_SYMBOLS
KEYBOARD_LOWER_CHOOSABLES = "`1234567890-=[]\;',./"
KEYBOARD_UPPER_CHOOSABLES = "~!@#$%^&*()_+{}|:\"<>?"



def apply_capitalization_to_char(new_char, caps_lock_is_on, shift_is_on):
    assert new_char in KEYBOARD_CHARS
    if new_char in KEYBOARD_LOWER_CHOOSABLES:
        if shift_is_on:
            new_char = KEYBOARD_UPPER_CHOOSABLES[KEYBOARD_LOWER_CHOOSABLES.index(new_char)]
    elif new_char in KEYBOARD_LETTERS:
        assert new_char.lower() == new_char, "assumption about keyboard abilities."
        if shift_is_on != caps_lock_is_on:
            new_char = new_char.upper()
    else:
        print("The character {} has unknown capitalization rules.".format(new_char))
    return new_char
    