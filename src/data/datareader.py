'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package.
@precondition: files must be available in given paths.
@summary: This file includes the functions to read grid, tweets and results of co-occurrence pattern analysis from files. 
'''

import logging
from data.datamodel import TweetGridMap, GridCell, KScoreAnalysis
from _collections import defaultdict
import cooc

def readTweetGridMaps(filename):
    logging.debug("Reading tweet grid maps")
    infile = open(filename, "r")
    lines = infile.readlines()
    infile.close()
    
    tweetGridMaps = []
    for line in lines:
        fields = unicode(line, 'utf-8').split('\t')
        tgm = TweetGridMap(int(fields[0]), float(fields[1]), float(fields[2]), fields[3].split())
        tweetGridMaps.append(tgm)
    logging.debug("Finished reading " + str(len(tweetGridMaps)) + " tweetgridmaps")
    return tweetGridMaps

def readGrid(filename):
    logging.debug("Reading grid")
    infile = open(filename, "r")
    lines = infile.readlines()
    infile.close()
    
    gridcells = []
    for line in lines:
        fields = line.split('\t')
        gc = GridCell(int(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]))
        gridcells.append(gc)
    logging.debug("Finished reading " + str(len(gridcells)) + " gridcells")
    return gridcells

def read_kscore_token_pairs(filename):
    logging.debug("Reading kscore token pairs")
    infile = open(filename, "r")
    lines = infile.readlines()
    infile.close()
    
    attraction_token_pairs_dict = defaultdict(set)
    repulsion_token_pairs_dict = defaultdict(set)
    for line in lines:
        fields = line.split('\t')
        kScoreAnalysis = KScoreAnalysis(unicode(fields[0], 'utf-8'), unicode(fields[1], 'utf-8'), int(fields[2]), float(fields[3]))
        if kScoreAnalysis.relationship == cooc.Relationship.ATTRACTION:
            attraction_token_pairs_dict[kScoreAnalysis.token_primary].add(kScoreAnalysis.token_pair)
        elif kScoreAnalysis.relationship == cooc.Relationship.REPULSION:
            repulsion_token_pairs_dict[kScoreAnalysis.token_primary].add(kScoreAnalysis.token_pair)
    logging.debug("There are " + str(len(attraction_token_pairs_dict)) + " primary tokens with relationship: " + str(cooc.Relationship.ATTRACTION))
    logging.debug("There are " + str(len(repulsion_token_pairs_dict)) + " primary tokens with relationship: " + str(cooc.Relationship.REPULSION))
    return attraction_token_pairs_dict, repulsion_token_pairs_dict

def read_bigrams(filename):
    logging.debug("Reading bigrams")
    infile = open(filename, "r")
    lines = infile.readlines()
    infile.close()
    
    token_pairs = set()
    for line in lines:
        fields = line.split('\t')
        kScoreAnalysis = KScoreAnalysis(unicode(fields[0], 'utf-8'), unicode(fields[1], 'utf-8'), int(fields[2]), float(fields[3]))
        token_pairs.add(kScoreAnalysis.token_pair)
    logging.debug("There are " + str(len(token_pairs)) + " distinct bigrams")
    return token_pairs
    
