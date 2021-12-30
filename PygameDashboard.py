


import time
import sys


import pygame


parent_module_exec = None

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
    

def stall_pygame():
    print("stall.")
    ENTER = 13
    BACKSPACE = 8
    CAPSLOCK = 1073741881
    SHIFT = 1073742053
    
    capsLockIsOn = False
    shiftIsOn = False
    
    commandString = ""
    running = True
    while running:
        time.sleep(0.1)
        pygame.display.flip()
        for event in pygame.event.get():
            #print((event, dir(event), event.type))
            if event.type in [pygame.K_ESCAPE, pygame.KSCAN_ESCAPE, pygame.QUIT]:
                print("closing pygame as requested.")
                pygame.display.quit()
                running = False
                break
                
            if event.type in [pygame.KEYDOWN, pygame.KEYUP]: # do this before checking key attribute because the event might not have one.
                if event.key == SHIFT:
                    if event.type == pygame.KEYDOWN:
                        shiftIsOn = True
                    if event.type == pygame.KEYUP:
                        shiftIsOn = False
                    continue
                
            if event.type in [pygame.KEYDOWN]:
                if event.key == ENTER:
                    print(__name__+">> "+commandString)
                    if parent_module_exec is None:
                        print("parent_module_exec was not provided. falling back to exec.")
                        exec(commandString)
                    else:
                        parent_module_exec(commandString)
                    commandString = ""
                elif event.key == BACKSPACE:
                    commandString = commandString[:-1]
                elif event.key == CAPSLOCK:
                    capsLockIsOn = not capsLockIsOn
                else:
                    try:
                        newChar = chr(event.key)
                        newChar = apply_capitalization_to_char(newChar, capsLockIsOn, shiftIsOn)
                        commandString = commandString + newChar
                    except Exception as e:
                        print("problem with char with code {}: {}.".format(event.key, e))
                        pass
                pygame.display.set_caption(commandString)
    print("end of stall.")