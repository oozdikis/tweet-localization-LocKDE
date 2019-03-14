'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package.
@precondition:
@summary: Definition of data models that represent entries in input files. 
'''

class TweetGridMap:
    def __init__(self, gcid, lat, lon, tokens):
        self.gcid = gcid
        self.lat = lat
        self.lon = lon
        self.tokens = tokens
        self.cartesian_coordinates = None
    
    def __str__(self):
        return str(self.tokens)      
    
    __repr__ = __str__

class GridCell:
    def __init__(self, gcid, latmin, lonmin, latmax, lonmax):
        self.gcid = gcid
        self.latmin = latmin
        self.lonmin = lonmin
        self.latmax = latmax
        self.lonmax = lonmax
    
    def __str__(self):
        return str(self.gcid)      
    
    __repr__ = __str__

class KScoreAnalysis:
    def __init__(self, token_primary, token_pair, relationship, kscore):
        self.token_primary = token_primary
        self.token_pair = token_pair
        self.relationship = relationship
        self.kscore = kscore
      
    def __str__(self):
        return "rel: " + str(self.relationship) + ", primary: " + self.token_primary + ", token_pair: " + self.token_pair + ", kscore: " + str(self.kscore)

    __repr__ = __str__