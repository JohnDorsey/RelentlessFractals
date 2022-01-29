


import time
import sys


import pygame

QUIT_EVENT_TYPES = {pygame.K_ESCAPE, pygame.KSCAN_ESCAPE, pygame.QUIT}


parent_module_exec = None

KEYBOARD_DIGITS = "0123456789"
KEYBOARD_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
KEYBOARD_SYMBOLS = " `~!@#$%^&*()-_=+[{]}\\|\\;:'\",<.>/?"
KEYBOARD_CHARS = KEYBOARD_DIGITS + KEYBOARD_LETTERS + KEYBOARD_SYMBOLS
KEYBOARD_LOWER_CHOOSABLES = "`1234567890-=[]\;',./"
KEYBOARD_UPPER_CHOOSABLES = "~!@#$%^&*()_+{}|:\"<>?"


def raise_inline(error_type, message):
    raise error_type(message)



def whichever_exec(command_string, preferred_exec=None):
    if preferred_exec is not None:
        execToUse = preferred_exec
    if parent_module_exec is not None:
        execToUse = parent_module_exec
    else:
        print("preferred or parent_module_exec was not provided. falling back to exec.")
        execToUse = exec
    execToUse(command_string)


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
    
    
   
   
def string_plus_pygame_key_event(string, event, caps_lock_is_on, shift_is_on):
    if event.key == pygame.K_BACKSPACE:
        return string[:-1]
        
    if not (0 <= event.key < 256):
        print("PygameDashboard.string_plus_pygame_key_event: ignoring extreme value {}.".format(event.key))
        return string
        
    try:
        newBaseChar = chr(event.key)
        newChar = apply_capitalization_to_char(newBaseChar, caps_lock_is_on, shift_is_on)
        result = string + newChar
        return result
    except Exception as e:
        print("PygameDashboard.string_plus_pygame_key_event: unexpected {} with char with code {}: {}.".format(type(e), event.key, e))
        return string





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
    def __init__(self, interval, min_rejections=None, max_rejections=None, cooldown_scale=0.0, require_finish_action=False):
        
        self.cooldown_scale = cooldown_scale
        self.last_action_duration = None
        self.require_finish_action = require_finish_action
        
        self.min_rejections, self.max_rejections = min_rejections, max_rejections
        self.rejection_counter = None
        
        self.interval = interval
        self.time_fun = time.monotonic
        self.last_approval_time = 0.0
        
        self._approve(self.time_fun())
        
    
    def get_effective_interval(self):
        return self.interval + (noneis(self.last_action_duration, 0.0) * self.cooldown_scale)
        
        
    def get_time_status(self):
        currentTime = self.time_fun()
        currentDuration = currentTime - self.last_approval_time
        remainingDuration = self.get_effective_interval() - currentDuration
        assert currentDuration > 0.0
        assert currentTime >= 0.0
        return (currentTime, currentDuration, remainingDuration)
        
        
    def _approve(self, approval_time):
        assert approval_time > self.last_approval_time
        self.last_approval_time = approval_time
        self.rejection_counter = 0
        self.last_action_duration = None
        
        
    def finish_action(self):
        assert self.last_action_duration is None, "self.last_action_duration not cleared. were methods called in bad order?"
        self.last_action_duration = self.time_fun() - self.last_approval_time
        
        
    def judge_by_max_rejection_count(self, current_time):
        if self.max_rejections is not None:
            if self.rejection_counter > self.max_rejections:
                self._approve(current_time)
                return True
            self.rejection_counter += 1
        return False
        
        
    def get_judgement(self):
        if self.require_finish_action:
            assert self.last_action_duration is not None, "finish_action was not called, but require_finish_action is True."
    
        if self.min_rejections is not None:
            if self.rejection_counter < self.min_rejections:
                self.rejection_counter += 1
                return False
        currentTime, _, timeRemaining = self.get_time_status()
        # assert timeElapsed > 0.0
        if timeRemaining <= 0.0:
            self._approve(currentTime)
            return True
        else:
            return self.judge_by_max_rejection_count(currentTime)
            
            
    def await_approval(self):
        currentTime, timeElapsed, timeRemaining = self.get_time_status()
        assert timeElapsed > 0.0
        
        if timeRemaining > 0.0:
            time.sleep(timeRemaining)

        self._approve(self.time_fun())
        return True
        



        
        

def is_quit_event(event):
    return event.type in QUIT_EVENT_TYPES
    
def clear_events_of_types(event_type_seq):
    for eventType in event_type_seq:
        pygame.event.clear(eventtype=eventType)
        

def stall_pygame(preferred_exec=None, clear_quit_events=True):
    print("stall.")
    
    keyTrack = KeyStateTracker([pygame.K_LSHIFT, pygame.K_CAPSLOCK], key_modes={pygame.K_LSHIFT:"is_down", pygame.K_CAPSLOCK:"odd_downs"})
    
    monitoredComStr = MonitoredValue("", pygame.display.set_caption)
    screenRateLimiter = RateLimiter(0.1)
    eventRateLimiter = RateLimiter(1/120.0)
    
    if clear_quit_events:
        clear_events_of_types(QUIT_EVENT_TYPES)
    
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
                whichever_exec(monitoredComStr.get(), preferred_exec=preferred_exec)
                monitoredComStr.set_sync("")
                continue
            
            if event.type == pygame.KEYDOWN:
                capslockState, lshiftState, rshiftState = keyTrack.get_translated_states([pygame.K_CAPSLOCK, pygame.K_LSHIFT, pygame.K_RSHIFT])
                shiftState = lshiftState or rshiftState
                monitoredComStr.set_sync(string_plus_pygame_key_event(monitoredComStr.get(), event, capslockState, shiftState))
                continue
    assert False