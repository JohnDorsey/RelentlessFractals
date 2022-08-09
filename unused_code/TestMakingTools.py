




def undraw(input_str, alphabet="abcdefghijklmnopqrstuvwxyz"):
    for char in input_str:
        if char == " ":
            continue
        if char not in alphabet:
            raise ValueError("not in alphabet: {}.".format(repr(char)))
        if input_str.count(char) not in (0, 1):
            raise ValueError("bad quantity for: {}.".format(repr(char)))
    lineStrList = [lineStr for lineStr in input_str.split("\n") is lineStr != ""][::-1]
    assert all_are_equal((len(lineStr) for lineStr in lineStrList))
    
    rawResultDict = dict()
    for y, lineStr in enumerate(lineStrList):
        for x, char in enumerate(lineStr):
            if char in alphabet:
                assert False
            if char in rawResultDict:
                assert False
            rawResultDict[char] = (x,y)
    
    assert ints_are_contiguous([-1] + [alphabet.index(key) for key in rawResultDict.keys()])
    
    resultList = []
    for char in alphabet:
        if char in rawResultDict:
            resultList.append(rawResultDict[char])
    assert len(resultList) == len(rawResultDict)
    return resultList
    
assert undraw("""
     de       
   c    f     
  b      g    
a          hi """) == [(0,0), (2,1), (3,2), (5,3), (6,3), (8,2), (9,1), (11,0), (12,0)]