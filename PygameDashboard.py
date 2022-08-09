#!/usr/bin/python3


import time
import copy

import Qwerty

import pygame




QUIT_EVENT_TYPES = {pygame.K_ESCAPE, pygame.KSCAN_ESCAPE, pygame.QUIT}

parent_module_exec = None



def assert_equal(thing0, thing1):
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)


def raise_inline(error_type, message):
    raise error_type(message)



def noneis(value, default):
    return default if value is None else value
    
def noneisnew(value, default_type):
    return default_type() if value is None else value

def assure_isinstance(value, required_type):
    assert isinstance(value, required_type), "could not assure value of type {} is instance of {}.".format(type(value), required_type)
    return value

def noneisnew_or_assure_isinstance(input_value, required_type):
    assert isinstance(required_type, type), raise_inline(ValueError, "required_type must be a type.")
    return assure_isinstance(noneisnew(input_value, required_type), required_type)

 




def whichever_exec(command_string, preferred_exec=None):
    if preferred_exec is not None:
        execToUse = preferred_exec
    if parent_module_exec is not None:
        execToUse = parent_module_exec
    else:
        print("preferred or parent_module_exec was not provided. falling back to exec.")
        execToUse = exec
    execToUse(command_string)


"""
def capture_exits(input_fun):
    def modifiedFun(*args, **kwargs):
        try:
            input_fun(*args, **kwargs)
        except KeyboardInterrupt:
            print("KeyboardInterrupt intercepted.")
            return
    return modifiedFun
"""





class SimpleLapTimer:
    def __init__(self):
        self.time_fun = time.monotonic
        self.last_press_time = None
    def peek(self):
        if self.last_press_time is None:
            return None
        return self.time_fun() - self.last_press_time
    def press(self):
        result = self.peek()
        self.last_press_time = self.time_fun()
        return result




def measure_time_nicknamed(nickname, end="\n", ndigits=(4,2,2), include_lap=False, include_load=False, _persistent_info=dict()): # copied from GeodeFractals/photo.py. heavily modified.
    if isinstance(ndigits, int):
        ndigitsList = [ndigits] * 3
    else:
        assert isinstance(ndigits, (list,tuple))
        assert len(ndigits) == 3
        ndigitsList = ndigits
    toMStr = lambda val: "{} m".format(round(val/60.0, ndigits=ndigitsList[1]))
    toHStr = lambda val: "{} h".format(round(val/60.0/60.0, ndigits=ndigitsList[2]))
    toSMHStr = lambda val: "{} s ({})({})".format(round(val, ndigits=ndigitsList[0]), toMStr(val), toHStr(val))
    toPercentStr = lambda val: "~{}%".format(round(100.0*val, ndigits=3))

    if not isinstance(nickname, str):
        raise TypeError("this decorator requires a string argument for a nickname to be included in the decorator line using parenthesis.")
        
    
    def measure_time_nicknamed_inner(input_fun):
        if not hasattr(input_fun, "__call__"):
            raise TypeError("this is not callable, and can't be decorated.")
            
        uidObj = object()
        assert uidObj not in _persistent_info
        localNickname = nickname
        if localNickname in (valB["nickname"] for valB in _persistent_info.values()):
            localNickname = localNickname + "({})".format(id(uidObj))
        _persistent_info[uidObj] = {"nickname":localNickname, "print_lap_info":include_lap, "print_load_info":include_load, "lap_end_time": None, "lap_count":0, "action_duration_sum":0.0, "lap_action_duration_sum":0.0, "lap_duration_sum":0.0}
        myInfo = _persistent_info[uidObj]
        del localNickname
        
        def measure_time_nicknamed_inner_inner(*args, **kwargs):
            actionStartTime = time.monotonic()
            result = input_fun(*args, **kwargs)
            actionEndTime = time.monotonic()
            
            actionDuration = actionEndTime - actionStartTime
            lapDuration = (actionEndTime - myInfo["lap_end_time"]) if myInfo["lap_end_time"] is not None else None
            
            print("{} took: {}.".format(nickname, toSMHStr(actionDuration)), end="")
            if myInfo["print_lap_info"]:
                print(" lap {} took: {}.".format(myInfo["lap_count"], toSMHStr(lapDuration) if lapDuration is not None else "unknown") if myInfo["print_lap_info"] else "", end="")
            if myInfo["print_load_info"]:
                print(" load: {}.".format(toPercentStr(actionDuration/lapDuration) if lapDuration is not None else "unknown"), end="")
            print("", end=end)
            
            myInfo["lap_end_time"] = actionEndTime
            myInfo["action_duration_sum"] += actionDuration
            if lapDuration is not None:
                myInfo["lap_duration_sum"] += lapDuration
                myInfo["lap_action_duration_sum"] += actionDuration
                myInfo["lap_count"] += 1
            
            return result
        return measure_time_nicknamed_inner_inner
        
    return measure_time_nicknamed_inner






        

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






class SimplifiedKeyEvent:
    def __init__(self, key, type_):
        self.key, self.type = (key, type_)


def string_plus_pygame_key_event(string, event, caps_lock_is_on=False, shift_is_on=False, suppress_warnings=False, forbid_warnings=False):
    if event.key == pygame.K_BACKSPACE:
        return string[:-1]
        
    if not (0 <= event.key < 256):
        if not suppress_warnings:
            print("PygameDashboard.string_plus_pygame_key_event: ignoring extreme value {}.".format(event.key))
            assert not forbid_warnings
        return string
        
    try:
        newBaseChar = chr(event.key)
        if newBaseChar in Qwerty.KEYBOARD_CHARS:
            newChar = Qwerty.apply_capitalization_to_char(newBaseChar, caps_lock_is_on, shift_is_on)
        else:
            if not suppress_warnings:
                print("PygameDashboard.string_plus_pygame_key_event: warning: char with code {} cannot be capitalized.".format(event.key))
                assert not forbid_warnings
            newChar = newBaseChar
        result = string + newChar
        return result
    except Exception as e:
        if isinstance(e, AssertionError):
            print("PygameDashboard.string_plus_pygame_key_event: AssertionError will be re-raised.")
            raise e
        print("PygameDashboard.string_plus_pygame_key_event: unexpected {} with char with code {}: {}.".format(type(e), event.key, e))
        assert not forbid_warnings
        return string
    assert False

assert string_plus_pygame_key_event("abcde", SimplifiedKeyEvent(pygame.K_BACKSPACE, pygame.KEYDOWN), False, False) == "abcd"





class KeyStateTracker:

    def __init__(self, key_codes, key_report_styles=None, key_registration_aliases=None):
        #  key_report_preprocessors=None
        self.key_codes = key_codes
        self.key_states = {code:{"is_down":False, "odd_downs":False, "odd_ups":False, "is_new":False} for code in self.key_codes}
        self.key_report_styles = noneisnew_or_assure_isinstance(key_report_styles, dict)
        self.key_registration_aliases = noneisnew_or_assure_isinstance(key_registration_aliases, dict)
        # self.key_report_preprocessors = key_report_preprocessors
        # self.key_code_aliases = noneisnew_or_assure_isinstance(key_code_aliases, dict)
        
    def register_event(self, event):
        if event.type not in [pygame.KEYDOWN, pygame.KEYUP]:
            return False
        if event.key not in self.key_codes:
            if event.key not in self.key_registration_aliases:
                return False
            else:
                alternativeKeyCode = self.key_registration_aliases[event.key]
                return self.register_event(SimplifiedKeyEvent(alternativeKeyCode, event.type))
                
        state = self.key_states[event.key]
        if event.type == pygame.KEYDOWN:
            state["is_down"] = True
            state["odd_downs"] = not state["odd_downs"]
        elif event.type == pygame.KEYUP:
            state["is_down"] = False
            state["odd_ups"] = not state["odd_ups"]
        else:
            assert False
        state["is_new"] = True
        return True
        
    def register_or_passthrough_event(self, event):
        success = self.register_event(event)
        if success:
            return None
        else:
            return event
            
    def register_or_passthrough_events(self, event_seq):
        for event in event_seq:
            currentResult = self.register_or_passthrough_event(event)
            if currentResult is not None:
                yield currentResult
    
    def get_state(self, key_code, peek=False):
        state = self.key_states[key_code]
        result = copy.deepcopy(state)
        if not peek:
            state["is_new"] = False
        return result
        
        
    def get_stylized_report(self, key_code, peek=False):
        if key_code not in self.key_report_styles:
            raise KeyError("no report style set for key with code {}.".format(key_code))
        else:
            return self.get_state(key_code, peek=peek)[self.key_report_styles[key_code]]
            
    def get_stylized_reports(self, key_codes, peek=False):
        return [self.get_stylized_report(keyCode, peek=peek) for keyCode in key_codes]
        
    
"""
def is_key_event(event):
    return event.type == test_type
"""

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
        













        
        

def is_quit_event(event):
    return event.type in QUIT_EVENT_TYPES
    
def clear_events_of_types(event_type_seq):
    for eventType in event_type_seq:
        pygame.event.clear(eventtype=eventType)
        

def gen_all_keydown_codes():
    raise NotImplementedError("deprecated")
    while True:
        for event in pygame.event.get():
            if is_quit_event(event):
                print("closing pygame display as requested.")
                pygame.display.quit()
                return
            if event.type == pygame.KEYDOWN:
                yield event.key



def gen_coin_sorter(input_seq, match_funs):
    raise NotImplementedError("not tested!")
    for item in input_seq:
        currentResult = [(item if matchFun(item) else None) for matchFun in match_funs]
        yield currentResult


def coin_handler(input_seq, method_pairs, match_multiple=True):
    raise NotImplementedError("not tested!")
    for item in input_seq:
        matchCount = 0
        for matchFun, handleMethod in fun_pairs:
            if matchFun(item):
                matchCount += 1
                handleMethod(item)
                if not match_multiple:
                    break
        else:
            if matchCount == 0:
                print("coin_handler: warning: unhandled item: {}.".format(item))





class PygameKeyboardPortableTranslator:
    def __init__(self, translatable_chars=Qwerty.KEYBOARD_CHARS, key_code_translations=None):
        self.translatable_chars = set(translatable_chars)
        self.key_code_translations = noneisnew_or_assure_isinstance(key_code_translations, dict)
        self.key_track = KeyStateTracker(
                [pygame.K_CAPSLOCK, "shift"],
                key_report_styles={"shift":"is_down", pygame.K_CAPSLOCK:"odd_downs"},
                key_registration_aliases={pygame.K_LSHIFT:"shift", pygame.K_RSHIFT:"shift"},
            )
            
    @property
    def capslock_state(self):
        return self.key_track.get_stylized_report(pygame.K_CAPSLOCK, peek=True)
    
    @property
    def shift_state(self):
        return self.key_track.get_stylized_report("shift", peek=True)
    
    def process_event(self, event, suppress_tracked=False):
        keyWasTracked = self.key_track.register_event(event)
        if keyWasTracked:
            if suppress_tracked:
                return None
            else:
                return event

        if event.type == pygame.KEYDOWN:
            if event.key in self.key_code_translations:
                charToUse = self.key_code_translations[event.key]
            elif chr(event.key) in self.translatable_chars:
                charToUse = chr(event.key)
            return Qwerty.apply_capitalization_to_char(charToUse, self.capslock_state, self.shift_state)

        return event
        
        

        
class PygameKeyboardPortablePrompt:
    def __init__(self):
        self.translator = PygameKeyboardPortableTranslator(key_code_translations={pygame.K_RETURN:"\n"})
        self.current_string = ""
        
    def process_event(self, raw_event):
        convertedEvent = self.translator.process_event(raw_event, suppress_tracked=True)
        
        
        if isinstance(convertedEvent, str):

            assert len(convertedEvent) == 1
            assert convertedEvent in Qwerty.KEYBOARD_CHARS
            if convertedEvent == "\b":
                self.current_string = self.current_string[:-1]
            else:
                self.current_string += convertedEvent
                assert "\b" not in self.current_string
        elif convertedEvent is None: # if was tracked and suppressed (if was capslock or shift):
            pass
        else:
            assert isinstance(convertedEvent, pygame.event.EventType), tpye(convertedEvent)
            assert not is_keydown_of(convertedEvent, pygame.K_RETURN), "key code translation should've caught this."

            if convertedEvent.type == pygame.KEYDOWN:
                assert convertedEvent.key not in Qwerty.KEYBOARD_ORDS
                self.current_string = string_plus_pygame_key_event(self.current_string, convertedEvent, self.translator.capslock_state, self.translator.shift_state, forbid_warnings=True)
        
        return convertedEvent
            
            
            
            
            

class PygameQuit(Exception):
    pass


portable_prompt = PygameKeyboardPortablePrompt()


def pygame_prompt(prompt_string="> ", preview_update_fun=pygame.display.set_caption, *, quit_return_value=None, default_string=None):
    assert portable_prompt.current_string == ""
    if default_string is not None:
        assert isinstance(default_string, str), type(default_string)
        portable_prompt.current_string = default_string

    screenRateLimiter = RateLimiter(0.1)
    eventRateLimiter = RateLimiter(1/120.0)
    
    while True:
        
        if screenRateLimiter.get_judgement():
            pygame.display.flip()
            
        eventRateLimiter.await_approval()
        
        # queuedKeyEvents.clear()
        
        # for event in pygameEventKeyboardTranslator.get_current_events():
        for rawEvent in pygame.event.get():
            if is_quit_event(rawEvent):
                portable_prompt.current_string = ""
                if quit_return_value is None:
                    raise PygameQuit()
                else:
                    return quit_return_value
                
            _ = portable_prompt.process_event(rawEvent)
            
            assert portable_prompt.current_string.count("\n") in (0,1)
            if portable_prompt.current_string.endswith("\n"):
                result = portable_prompt.current_string[:-1]
                portable_prompt.current_string = ""
                preview_update_fun("")
                return result
            else:
                preview_update_fun(prompt_string + portable_prompt.current_string)
                
    assert False
    

def stall_pygame(prompt_string="> ", preview_update_fun=pygame.display.set_caption, preferred_exec=None, clear_quit_events=True, autostart_display=(128,128)):
    """
        preferred_exec: the function that will be called to execute a command once it ends with a newline.
    """
    print("stall.")
    
    if autostart_display is not None:
        if pygame.display.get_surface() is None:
            pygame.display.set_mode(autostart_display)
            
    if clear_quit_events:
        clear_events_of_types(QUIT_EVENT_TYPES)
    
    
    while True:
        try:
            comStr = pygame_prompt(prompt_string=prompt_string, preview_update_fun=preview_update_fun)
            whichever_exec(comStr, preferred_exec=preferred_exec)
            portable_prompt.current_string = ""
            continue
        except PygameQuit:
            print("closing pygame display as requested.")
            pygame.display.quit()
            return
    

if __name__ == "__main__":
    print("running PygameDashboard demo: stall_pygame.")
    stall_pygame()