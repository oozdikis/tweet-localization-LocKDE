'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package. 
@precondition: data.grid_file, data.training_file and data.test_file must be available in configured paths.
@summary: This includes the main function to perform prediction. It creates a LocKDE-SCoP classifier and runs the tests on test data. 
'''

import logging

from data import datareader
import logger_settings
from prediction import cooc_feature_space, classifier
import data
from geopy.distance import vincenty
import numpy as np

def main():
    logger_settings.setLoggers(None)

    grid = datareader.readGrid(data.grid_file)
    tgms_training = datareader.readTweetGridMaps(data.training_file)
    tgms_test = datareader.readTweetGridMaps(data.test_file)
    tokens_set = set()
    for tgm in tgms_training:
        tokens_set = tokens_set.union(set(tgm.tokens))
    tokens_list = list(tokens_set)
    tokens_list.sort()
    tokens_dictionary = {}
    for i in range(len(tokens_list)):
        token = tokens_list[i]
        tokens_dictionary[token] = i
    cooc_feature_space.extend_feature_space_with_bigrams(tgms_training, tgms_test, tokens_set)
    tgms_training = get_tgms_with_tokens(tgms_training, tokens_set)
    logging.info('Running predictions using ' + str(len(tokens_set)) + ' features in the training set')
    
    kdeClassifier = classifier.KDESum(grid, tgms_training, tokens_set)
    logging.info("classify_test_tweets_using_KDE for %d test items" %len(tgms_test))
    tgms_test_predicted_gcid_dict = {}
    for i in range(len(tgms_test)):
        tgm_test = tgms_test[i]
        predicted_gcid = kdeClassifier.predictLocation(set(tgm_test.tokens))
        tgms_test_predicted_gcid_dict[i] = predicted_gcid
        
    errorDistancesInMeters = []
    for i in range(len(tgms_test)):
        tgm_test = tgms_test[i]
        predicted_gcid = tgms_test_predicted_gcid_dict[i]
        predlat = (grid[predicted_gcid].latmin + grid[predicted_gcid].latmax) / 2.0
        predlon = (grid[predicted_gcid].lonmin + grid[predicted_gcid].lonmax) / 2.0 
        distanceInMeters = float(vincenty((tgm_test.lat, tgm_test.lon), (predlat, predlon)).meters)
        errorDistancesInMeters.append(distanceInMeters)
    logging.info("medianErrorDistanceInMeters: " + str(np.median(errorDistancesInMeters)))

def get_tgms_with_tokens(tgms, tokens):
    tgms_with_frequent_tokens = []
    for tgm in tgms:
        found_tokens_in_tweet = [token for token in tgm.tokens if token in tokens]
        if len(found_tokens_in_tweet) > 0:
            tgm.tokens = found_tokens_in_tweet
            tgms_with_frequent_tokens.append(tgm)
    logging.debug("Found " + str(len(tgms_with_frequent_tokens)) + " tweetgridmaps with tokens.")
    return tgms_with_frequent_tokens    
        
if __name__ == '__main__':
    main()
