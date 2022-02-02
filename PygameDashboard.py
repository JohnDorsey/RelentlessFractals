


import time
import sys

import Qwerty

import pygame

QUIT_EVENT_TYPES = {pygame.K_ESCAPE, pygame.KSCAN_ESCAPE, pygame.QUIT}


parent_module_exec = None



def assert_equal(thing0, thing1):
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)


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


    
   
   
def string_plus_pygame_key_event(string, event, caps_lock_is_on, shift_is_on):
    if event.key == pygame.K_BACKSPACE:
        return string[:-1]
        
    if not (0 <= event.key < 256):
        print("PygameDashboard.string_plus_pygame_key_event: ignoring extreme value {}.".format(event.key))
        return string
        
    try:
        newBaseChar = chr(event.key)
        newChar = Qwerty.apply_capitalization_to_char(newBaseChar, caps_lock_is_on, shift_is_on)
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
        
        
        

class SimpleRateLimiter:
    def __init__(self, interval, cooldown_scale=0.0, require_finish_action=False):
        
        self.cooldown_scale = cooldown_scale
        self.last_action_duration = 0.0
        self.require_finish_action = require_finish_action
        
        self.interval = interval
        self.time_fun = time.monotonic
        self.last_approval_time = 0.0
        
        self._approve(self.time_fun())
        
        
    def _validate_action_finished_state(self):
        if self.require_finish_action:
            assert self.last_action_duration is not None, "finish_action was not called, but require_finish_action is True."
        
        
    def _approve(self, approval_time):
        self._validate_action_finished_state()
        assert approval_time > self.last_approval_time
        self.last_approval_time = approval_time
        self.last_action_duration = None
        
        
    def _reject(self):
        self._validate_action_finished_state()
        
    
    def _apply_judgement(self, judgement, current_time):
        if judgement:
            self._approve(current_time)
        else:
            self._reject()
        
        
    def get_effective_interval(self):
        return self.interval + (noneis(self.last_action_duration, 0.0) * self.cooldown_scale)
        
        
    def get_time_status(self):
        currentTime = self.time_fun()
        currentDuration = currentTime - self.last_approval_time
        remainingDuration = self.get_effective_interval() - currentDuration
        assert currentDuration > 0.0
        assert currentTime >= 0.0
        return (currentTime, currentDuration, remainingDuration)
        
        
    def finish_action(self):
        assert self.last_action_duration is None, "self.last_action_duration not cleared. were methods called in bad order?"
        self.last_action_duration = self.time_fun() - self.last_approval_time
        
        
    def _peek_judgement_by_time(self, current_time, time_remaining):
        return (time_remaining <= 0.0)
        
        
    def get_judgement(self, peek=False):
        currentTime, _, timeRemaining = self.get_time_status()
        result = self._peek_judgement_by_time(currentTime, timeRemaining)
        if not peek:
            self._apply_judgement(result, currentTime)
        return result
            
            
    def await_approval(self):
        _, _, timeRemaining = self.get_time_status()
        
        if timeRemaining > 0.0:
            time.sleep(timeRemaining)

        self._approve(self.time_fun())
        return True


testA = SimpleRateLimiter(0.1)
testB = SimpleRateLimiter(0.1)
testA._approve(testA.time_fun())
testB._approve(testB.time_fun())
assert testA.last_approval_time < testB.last_approval_time
testA._approve(testA.time_fun())
assert testB.last_approval_time < testA.last_approval_time
assert testA.get_judgement() == False
time.sleep(testA.interval)
assert testA.get_judgement() == True
assert testA.get_judgement() == False
del testA
del testB


class RateLimiter(SimpleRateLimiter):
    def __init__(self, interval, min_rejections=None, max_rejections=None, cooldown_scale=0.0, require_finish_action=False):
        self.min_rejections, self.max_rejections = (min_rejections, max_rejections)
        self.rejection_counter = 0
        assert isinstance(cooldown_scale, float)
        super().__init__(interval, cooldown_scale=cooldown_scale, require_finish_action=require_finish_action)
        
        
    def _approve(self, approval_time):
        super()._approve(approval_time)
        self.rejection_counter = 0
        
        
    def _reject(self):
        super()._reject()
        self.rejection_counter += 1
        
        
    def _peek_judgement_by_max_rejection_count(self):
        return ((self.max_rejections is not None) and (self.rejection_counter >= self.max_rejections))
        
        
    def _peek_judgement(self):
        if (self.min_rejections is not None) and (self.rejection_counter < self.min_rejections):
            return False
        else:
            if super().get_judgement(peek=True):
                return True
            if self._peek_judgement_by_max_rejection_count():
                return True
            return False
            
            
    def get_judgement(self, peek=False):
        result = self._peek_judgement()
        if not peek:
            self._apply_judgement(result, self.time_fun())
        return result
        
                
testA = RateLimiter(0.0, min_rejections=3)
assert_equal([testA.get_judgement() for i in range(10)], [False, False, False, True, False, False, False, True, False, False])
testA = RateLimiter(0.1, min_rejections=3)
assert_equal([testA.get_judgement() for i in range(10)], [False] * 10)

testA = RateLimiter(0.2, max_rejections=2)
assert_equal([testA.get_judgement() for i in range(10)], [False, False, True, False, False, True, False, False, True, False])
testA.await_approval()
assert_equal([testA.get_judgement() for i in range(10)], [False, False, True, False, False, True, False, False, True, False])
del testA


        
        

def is_quit_event(event):
    return event.type in QUIT_EVENT_TYPES
    
def clear_events_of_types(event_type_seq):
    for eventType in event_type_seq:
        pygame.event.clear(eventtype=eventType)
        





def stall_pygame(preferred_exec=None, clear_quit_events=True):
    print("stall.")
    
    keyTrack = KeyStateTracker([pygame.K_CAPSLOCK, pygame.K_LSHIFT, pygame.K_RSHIFT], key_modes={pygame.K_LSHIFT:"is_down", pygame.K_RSHIFT:"is_down", pygame.K_CAPSLOCK:"odd_downs"})
    
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
                print("closing pygame display as requested.")
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