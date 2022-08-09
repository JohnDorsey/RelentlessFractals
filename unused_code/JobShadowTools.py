


# ExplicitPartial = namedtuple("ExplicitPartialNamedTuple", ["function", "args", "kwargs"])


class FuncJobShadowUnit:
    def __init__(self, starting_members):
        self.members = [item for item in starting_members]
        
    def add(self, fun):
        # funEPNT = fun if isinstance(fun, ExplicitPartial) else ExplicitPartial(fun, list(), dict())
        # assert len(funEPNT) == 2
        # self.members.append(funEPNT)
        if fun in self.members:
            raise ValueError("already included.")
        self.members.append(fun)
        
    def __call__(self, *args, **kwargs):
        # resultsGen = (currentFunTup[0](*args, *currentFunTup.args, **currentFunTup.kwargs, **kwargs) for currentFunTup in self.members)
        resultsGen = (currentFun(*args, **kwargs) for currentFun in self.members)
        return get_shared_value(resultsGen)
        
    #def contains_variant_of(self, fun):
    # ...   
    

class FuncJobShadowFloor:
    def __init__(self):
        self.units = list()
        
    # def get_unit(self, fun):
    #     return
    
    def register(self, new_group):
        """
        for newFun in new_group:
            for existingUnit in self.units:
                if newFun in existingUnit:
                    raise ValueError("already registered.")
        """
        raise NotImplementedError()
        
        
        
        
        