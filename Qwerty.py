

# import unicodedata

KEYBOARD_DIGITS = "0123456789"
KEYBOARD_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
KEYBOARD_SYMBOLS = " `~!@#$%^&*()-_=+[{]}\\|\\;:'\",<.>/?"
KEYBOARD_CONTROL_CHARS = "\n\t\b"
KEYBOARD_CHARS = KEYBOARD_DIGITS + KEYBOARD_LETTERS + KEYBOARD_SYMBOLS + KEYBOARD_CONTROL_CHARS
KEYBOARD_ORDS = [ord(char) for char in KEYBOARD_CHARS]

KEYBOARD_LOWER_CHOOSABLES = "`1234567890-=[]\;',./"
KEYBOARD_UPPER_CHOOSABLES = "~!@#$%^&*()_+{}|:\"<>?"

LMMS_PIANO_NOTE_ROW_OPTIONS = ["zsxdcvgbhnjm,l.;/               ", # rows MUST be the same length for code in this file!
                               "            q2w3er5t6y7ui9o0p[=]"]
LMMS_PIANO_NOTE_KEY_COUNT = len(LMMS_PIANO_NOTE_ROW_OPTIONS[0])
LMMS_PIANO_NOTE_KEY_OPTIONS = [tuple(noteRow[noteIndex] for noteRow in LMMS_PIANO_NOTE_ROW_OPTIONS if noteRow[noteIndex] != " ") for noteIndex in range(LMMS_PIANO_NOTE_KEY_COUNT)]
LMMS_PIANO_NOTE_READABLE_KEYS = [sorted(LMMS_PIANO_NOTE_KEY_OPTIONS[noteIndex], key=(lambda testOption: (KEYBOARD_LETTERS+KEYBOARD_DIGITS+KEYBOARD_SYMBOLS).index(testOption)))[0] for noteIndex in range(LMMS_PIANO_NOTE_KEY_COUNT)]
LMMS_PIANO_NOTE_CHAR_SET = set.union(*[set(char for char in noteStr if char != " ") for noteStr in LMMS_PIANO_NOTE_ROW_OPTIONS])


def apply_capitalization_to_char(new_char, caps_lock_is_on, shift_is_on):
    if not new_char in KEYBOARD_CHARS:
        raise ValueError("char {} (ord {}) is not in KEYBOARD_CHARS.".format(new_char, ord(new_char)))
    if new_char in KEYBOARD_LOWER_CHOOSABLES:
        if shift_is_on:
            new_char = KEYBOARD_UPPER_CHOOSABLES[KEYBOARD_LOWER_CHOOSABLES.index(new_char)]
    elif new_char in KEYBOARD_LETTERS:
        assert new_char.lower() == new_char, "assumption about keyboard abilities."
        if shift_is_on != caps_lock_is_on:
            new_char = new_char.upper()
    elif new_char in KEYBOARD_CONTROL_CHARS:
        pass
    else:
        print("The character {} (ord {}) has unknown capitalization rules.".format(new_char, ord(new_char)))
    return new_char
    