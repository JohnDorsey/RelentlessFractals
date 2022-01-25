


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


def raise_inline(error_type, message):
    raise error_type(message)



def whichever_exec(command_string):
    if parent_module_exec is None:
        print("parent_module_exec was not provided. falling back to exec.")
        exec(command_string)
    else:
        parent_module_exec(command_string)


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
    
    
def string_plus_char(string=None, new_char=None):
    if new_char == char(pygame.K_BACKSPACE):
        return string[:-1]
    else:
        return string + new_char
   
   
def string_plus_pygame_key_event(string, event, caps_lock_is_on, shift_is_on):
    try:
        newBaseChar = chr(event.key)
        newChar = apply_capitalization_to_char(newBaseChar, caps_lock_is_on, shift_is_on)
        result = string_plus_char(string=string, new_char=newChar)
        return result
    except Exception as e:
        print("problem with char with code {}: {}.".format(event.key, e))
        return string


def is_quit_event(event):
    return event.type in [pygame.K_ESCAPE, pygame.KSCAN_ESCAPE, pygame.QUIT]
   

def noneis(value, default):
    return default if value is None else value
    
def noneisnew(value, default_type):
    return default_type() if value is None else value

def assure_isinstance(value, required_type):
    assert isinstance(value, required_type), "could not assure value of type {} is instance of {}.".format(type(input_value), required_type)
    return value

def noneisnew_or_assure_isinstance(input_value, required_type):
    assert isinstance(required_type, type), raise_inline(ValueError, "required_type must be a type.")
    return assure_isinstance(noneisnew(input_value, required_type), required_type)

"""
def noneisnew_or_assure_isinstance(input_value, required_type):
    result = required_type() if input_value is None else input_value
    assert isinstance(result, required_type), "with input of type {}, could not assure result of type {} is instance of {}.".format(type(input_value), type(result), required_type)
    return result
"""    


class KeyStateTracker:
    def __init__(self, key_codes, key_modes=None):
        self.key_codes = key_codes
        self.key_states = {code:{"is_down":False, "odd_downs":False, "odd_ups":False} for code in self.key_codes}
        self.key_modes = noneisnew_or_assure_isinstance(key_modes, dict)
        # self.key_code_aliases = noneisnew_or_assure_isinstance(key_code_aliases, dict)
        
    def register(self, event):
        if event.type not in [pygame.KEYDOWN, pygame.KEYUP]:
            return False
        if event.key not in self.key_codes:
            return False
        state = self.key_states[event.key]
        if event.type == pygame.KEYDOWN:
            state["is_down"] = True
            state["odd_downs"] = not state["odd_downs"]
        elif event.type == pygame.KEYUP:
            state["is_down"] = False
            state["odd_ups"] = not state["odd_ups"]
        else:
            assert False
        return True
    
    def get_state(self, key_code):
        return self.key_states[key_code]
        
    def get_translated_state(self, key_code):
        if key_code not in self.key_modes:
            raise KeyError("no mode set for key with code {}.".format(key_code))
        else:
            return self.get_state(key_code)[self.key_modes[key_code]]
            
    def get_translated_states(self, key_codes):
        return [self.get_translated_state(keyCode) for keyCode in key_codes]
        
    
def is_exact_key_event(event, test_type, key_code):
    if event.type == test_type:
        if event.key == key_code:
            return True
    return False
    
def is_keydown_of(event, key_code):
    return is_exact_key_event(event, pygame.KEYDOWN, key_code)

def is_keyup_of(event, key_code):
    return is_exact_key_event(event, pygame.KEYUP, key_code)


class MonitoredValue:
    def __init__(self, initial_value, sync_callback):
        self.latest_value = initial_value
        self.synced_value = None
        self.sync_callback = sync_callback

    def set_async(self, value):
        self.latest_value = value
        
    def set_sync(self, value):
        self.set_async(value)
        self.resync()
        
    def resync(self):
        if self.latest_value == self.synced_value:
            return (False, None)
        else:
            result = self.sync_callback(self.latest_value)
            self.synced_value = self.latest_value
            return (True, result)
            
    def get(self):
        return self.latest_value
        
    def get_synced(self):
        return self.synced_value
        
        
class RateLimiter:
    def __init__(self, interval, min_rejections=None, max_rejections=None):
        self.interval = interval
        self.min_rejections, self.max_rejections = min_rejections, max_rejections
        self.time_fun = time.monotonic
        
        self.last_approval_time, self.rejection_counter = (0.0, None)
        self.approve(self.time_fun())
        
    def approve(self, approval_time):
        assert approval_time > self.last_approval_time
        self.last_approval_time = approval_time
        self.rejection_counter = 0
        
    def judge_by_rejection_count(self, current_time):
        if self.max_rejections is not None:
            if self.rejection_counter > self.max_rejections:
                self.approve(current_time)
                return True
            self.rejection_counter += 1
        return False
        
    def get_judgement(self):
        if self.min_rejections is not None:
            if self.rejection_counter < self.min_rejections:
                self.rejection_counter += 1
                return False
        currentTime = self.time_fun()
        actualInterval = currentTime - self.last_approval_time
        assert actualInterval > 0.0
        if actualInterval >= self.interval:
            self.approve(currentTime)
            return True
        else:
            result = self.judge_by_rejection_count(currentTime)
            return result
            
    def await_approval(self):
        currentTime = self.time_fun()
        currentDuration = currentTime - self.last_approval_time
        assert currentDuration > 0.0
        remainingDuration = self.interval - currentDuration
        if remainingDuration <= 0.0:
            self.approve(currentTime)
            return True
        else:
            time.sleep(remainingDuration)
            self.approve(self.time_fun())
            return True
        

def stall_pygame():
    print("stall.")
    
    keyTrack = KeyStateTracker([pygame.K_LSHIFT, pygame.K_CAPSLOCK], key_modes={pygame.K_LSHIFT:"is_down", pygame.K_CAPSLOCK:"odd_downs"})
    
    monitoredComStr = MonitoredValue("", pygame.display.set_caption)
    screenRateLimiter = RateLimiter(0.1)
    eventRateLimiter = RateLimiter(0.01)
    
    while True:
        
        if screenRateLimiter.get_judgement():
            pygame.display.flip()
            
        eventRateLimiter.await_approval()
        
        for event in pygame.event.get():
        
            if is_quit_event(event):
                print("closing pygame as requested.")
                pygame.display.quit()
                return
                
            keyWasTracked = keyTrack.register(event)
            if keyWasTracked:
                continue
                
            if is_keydown_of(event, pygame.K_RETURN):
                print(__name__+">> "+monitoredComStr.get())
                whichever_exec(monitoredComStr.get())
                monitoredComStr.set_sync("")
                continue
            
            if event.type == pygame.KEYDOWN:
                monitoredComStr.set_sync(string_plus_pygame_key_event(monitoredComStr.get(), event, *keyTrack.get_translated_states(pygame.K_CAPSLOCK, pygame.K_LSHIFT)))
                continue
    assert False