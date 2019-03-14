'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package.
@precondition: data.training_file and data.kscore_analysis_file must be available in configured paths.
@summary: This file includes the class definition that performs prediction using KDE. 
Training is performed during the initialization of an instance.
predictLocation() can be called after the training (initialization) is finished. 
'''

import logging

from _collections import defaultdict
import numpy as np
import warnings
from prediction import information_gain_ratio, cooc_feature_space
import operator
from data import datareader
import data
from scipy import stats
from numpy.core.shape_base import atleast_2d
from numpy import linalg

SCOTT_CONSTANT = -0.166666666666667

class KDESum:
    def __init__(self, grid, tgms_training, tokens_set):
        logging.debug("init KDE_SUM")
        warnings.simplefilter("error", RuntimeWarning)
        self.grid = grid
        self.tokens_set = tokens_set
        self.inf_gain_ratios = self.get_inf_gain_ratios()
        token_with_min_igr = min(self.inf_gain_ratios, key = lambda x: self.inf_gain_ratios.get(x) )
        self.min_inf_gain_ratio_score = self.inf_gain_ratios[token_with_min_igr]
        self.token_tgms_mappings = defaultdict(list)
        for tgm in tgms_training:
            tgm_tokens_set = set(tgm.tokens)
            for token in tgm_tokens_set:
                self.token_tgms_mappings[token].append(tgm)
        self.grid_tgm_counts = defaultdict(int)
        for tgm in tgms_training:
            self.grid_tgm_counts[tgm.gcid] += 1
        self.gcid_with_max_prior = max(self.grid_tgm_counts.iteritems(), key=operator.itemgetter(1))[0]
        tokens_list = list(self.tokens_set)
        list.sort(tokens_list)
        self.gc_probabilities_dict_for_tokens = {}
        logging.debug("Finding prior probabilities for tokens...")
        for token in tokens_list:
            observation_coordinates = self.get_observation_coordinates_of_token(token)
            if len(observation_coordinates[0]) < 2:
                continue
            bw = pow(len(observation_coordinates[0]), SCOTT_CONSTANT) * (( 1.0 - self.inf_gain_ratios[token]) + self.min_inf_gain_ratio_score)
            kernel = stats.kde.gaussian_kde(observation_coordinates, bw)
            self.gc_probabilities_dict_for_tokens[token] = self.get_probability_assignments(kernel, grid)
        logging.debug("Initialization finished")
        
    def get_inf_gain_ratios(self):
        unigram_inf_gain_ratios = self.get_unigram_inf_gain_ratios()
        bigram_inf_gain_ratios = self.get_bigram_inf_gain_ratios()
        inf_gain_ratios = dict(unigram_inf_gain_ratios.items() + bigram_inf_gain_ratios.items())
        return inf_gain_ratios

    def get_unigram_inf_gain_ratios(self):
        tgms_training = datareader.readTweetGridMaps(data.training_file)
        tokens_set = set()
        for tgm in tgms_training:
            tokens_set = tokens_set.union(set(tgm.tokens))
        tokens_list = list(tokens_set)
        unigram_inf_gain_ratios = information_gain_ratio.find_inf_gain_ratios(tgms_training, tokens_list)        
        return unigram_inf_gain_ratios
    
    def get_bigram_inf_gain_ratios(self):
        tgms_training = datareader.readTweetGridMaps(data.training_file)
        bigrams_set = datareader.read_bigrams(data.kscore_analysis_file)
        cooc_feature_space.generate_bigrams_from_unigrams(tgms_training, bigrams_set)
        tgms_training = [tgm for tgm in tgms_training if len(tgm.tokens) > 0]
        tokens_set = set()
        for tgm in tgms_training:
            tokens_set = tokens_set.union(set(tgm.tokens))
        tokens_list = list(tokens_set)
        bigram_inf_gain_ratios = information_gain_ratio.find_inf_gain_ratios(tgms_training, tokens_list)        
        return bigram_inf_gain_ratios    
    
    def get_observation_coordinates_of_token(self, token):
        tgms_of_token = self.token_tgms_mappings[token]
        observation_coordinates = [[],[]]
        if len(tgms_of_token) < 2:
            return observation_coordinates
        for tgm in tgms_of_token:
            observation_coordinates[0].append(tgm.lat)
            observation_coordinates[1].append(tgm.lon)
        _cov = atleast_2d(np.cov(observation_coordinates, rowvar=1, bias=False))
        _det = linalg.det(_cov)
        if _det <= 0:
            observation_coordinates = self.tune_observation_coordinates(observation_coordinates)
        try:
            bw = pow(len(observation_coordinates[0]), SCOTT_CONSTANT) * (( 1.0 - self.inf_gain_ratios[token]) + self.min_inf_gain_ratio_score)
            stats.kde.gaussian_kde(observation_coordinates, bw)
        except RuntimeWarning:
            observation_coordinates = self.tune_observation_coordinates(observation_coordinates)
        return observation_coordinates
    
    def get_probability_assignments(self, kernel, grid):
        gc_probabilities_dict = defaultdict(float)
        for gridcell in grid:
            gc_probability = kernel.integrate_box((gridcell.latmin, gridcell.lonmin), (gridcell.latmax, gridcell.lonmax))
            gc_probabilities_dict[gridcell.gcid] = np.nan_to_num(gc_probability)
        return gc_probabilities_dict
    
    def predictLocation(self, tokens_in_tweet):
        gc_probabilities_for_tweet = defaultdict(float)
        token_found = False
        for token in tokens_in_tweet:
            gc_probabilities_dict_for_token = self.gc_probabilities_dict_for_tokens.get(token)
            if gc_probabilities_dict_for_token == None:
                continue
            else:
                token_found = True
                for gcid in gc_probabilities_dict_for_token.keys():
                    gc_probability_for_token = gc_probabilities_dict_for_token[gcid]
                    gc_probabilities_for_tweet[gcid] += gc_probability_for_token * self.inf_gain_ratios[token]
        if token_found:
            return max(gc_probabilities_for_tweet.iteritems(), key=operator.itemgetter(1))[0]
        else:
            return self.gcid_with_max_prior    
    
    def tune_observation_coordinates(self, lats_and_lons):
        latlon_counts = defaultdict(int)
        for i in range(len(lats_and_lons[0])):
            latlon_counts[(lats_and_lons[0][i],lats_and_lons[1][i])] += 1
        latlon = max(latlon_counts.iteritems(), key=operator.itemgetter(1))[0]
        count = len(lats_and_lons[0])
        for i in reversed(range(count)):
            lat = lats_and_lons[0][i]
            lon = lats_and_lons[1][i]
            if (lat == latlon[0]) and (lon == latlon[1]):
                lats_and_lons[0].pop(i)
                lats_and_lons[1].pop(i)
            if count - len(lats_and_lons[0]) == 4:
                break
        lats_and_lons[0].extend([latlon[0] - 0.001, latlon[0] - 0.001, latlon[0] + 0.001, latlon[0] + 0.001])
        lats_and_lons[1].extend([latlon[1] - 0.001, latlon[1] + 0.001, latlon[1] - 0.001, latlon[1] + 0.001])
        return lats_and_lons
