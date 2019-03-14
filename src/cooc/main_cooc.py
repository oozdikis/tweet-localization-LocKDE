'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package. 
@precondition: data.training_file and data.grid_file must be available in configured paths.
@summary: This code finds the bigrams with attraction-repulsion patterns in training data and writes them to the file at data.kscore_analysis_file.    
'''

import logging
import logger_settings
from data import datareader
import data
import cooc
import numpy as np
from geopy.distance import great_circle
from cooc import ripley_k_function
import math

def main():
    logger_settings.setLoggers()
    grid = datareader.readGrid(data.grid_file)
    tgms_training = datareader.readTweetGridMaps(data.training_file)
    tokens_set = set()
    for tgm in tgms_training:
        tokens_set = tokens_set.union(set(tgm.tokens))
    tokens_list = list(tokens_set)
    tokens_list.sort()
    
    tokens_dictionary = {}
    for i in range(len(tokens_list)):
        token = tokens_list[i]
        tokens_dictionary[token] = i
    find_attration_repulsion_pairs(grid, tokens_dictionary, tgms_training, tokens_list)
    logging.info("Finished finding attraction-repulsion bigrams")  

def find_attration_repulsion_pairs(grid, tokens_dictionary, tgms_training, tokens_list):
    logging.info("Starting find_attration_repulsion_pairs. KFUNCTION_DELTA_DISTANCE_KM: " + str(cooc.KFUNCTION_DELTA_DISTANCE_KM) + ", N_MONTE_CARLO_SIMULATIONS: " + str(cooc.N_MONTE_CARLO_SIMULATIONS))
    sparse_rowArr, sparse_colArr = get_sparse_feature_matrix(tokens_dictionary, tgms_training)
    area = get_grid_area(grid)
    for tgm in tgms_training:
        tgm.cartesian_coordinates = ripley_k_function.latlon_to_cartesian(tgm.lat,tgm.lon) 
    delta_distance_cartesian = ripley_k_function.convert_distance_to_cartesian_at_point(tgms_training[0].lat, tgms_training[0].lon, cooc.KFUNCTION_DELTA_DISTANCE_KM)

    f = open(data.kscore_analysis_file, "w")
    counter = 0
    for token_primary in tokens_list:
        counter += 1
        logging.debug("Analyzing relationships for primary_token: " + token_primary + " (" + str(counter) + "/" + str(len(tokens_list)) + ")")
        kScoreAnalyses = ripley_k_function.analyze_relationships_of_token(grid, tokens_dictionary, tgms_training, sparse_rowArr, sparse_colArr, token_primary, delta_distance_cartesian, area)
        for kScoreAnalysis in kScoreAnalyses:
            logging.debug(kScoreAnalysis)
            f.write(kScoreAnalysis.token_primary.encode('utf-8') + "\t" + kScoreAnalysis.token_pair.encode('utf-8') + "\t" + str(kScoreAnalysis.relationship) + "\t" + str(kScoreAnalysis.kscore) + "\n")
            f.flush()
    f.close()

def get_sparse_feature_matrix(tokens_dictionary, tgms):
    rowArr = []
    colArr = []
    tweetCounter = 0
    for tgm in tgms:
        unique_tokens_in_tweet = []
        for token in tgm.tokens:
            if token not in unique_tokens_in_tweet:
                unique_tokens_in_tweet.append(token)
        for tokenInTweet in unique_tokens_in_tweet:
            rowArr.append(tweetCounter)
            tokenIndex = tokens_dictionary[tokenInTweet]
            colArr.append(tokenIndex)
        tweetCounter += 1
    rowArr = np.array(rowArr)
    colArr = np.array(colArr)
    return rowArr, colArr


def get_grid_area(grid):
    grid_latmin, grid_lonmin = grid[0].latmin, grid[0].lonmin
    grid_latmax, grid_lonmax = grid[-1].latmax, grid[-1].lonmax
    x_distance = float(great_circle((grid_latmin, grid_lonmin), (grid_latmin, grid_lonmax)).kilometers)
    y_distance = float(great_circle((grid_latmin, grid_lonmin), (grid_latmax, grid_lonmin)).kilometers)
    initial_area = x_distance * y_distance
    sw_x, _, _ = ripley_k_function.latlon_to_cartesian(grid_latmin, grid_lonmin) 
    se_x, _, _ = ripley_k_function.latlon_to_cartesian(grid_latmin, grid_lonmax)
    nw_x, _, _ = ripley_k_function.latlon_to_cartesian(grid_latmax, grid_lonmin) 
    ne_x, _, _ = ripley_k_function.latlon_to_cartesian(grid_latmax, grid_lonmax)
    x_length_south = math.fabs(sw_x - se_x)
    x_length_north = math.fabs(nw_x - ne_x)
    (longer_x, shorter_x) = (x_length_south, x_length_north) if x_length_south >= x_length_north else (x_length_north, x_length_south)
    x_diff = longer_x - shorter_x
    earth_area_ratio = 0.5 * (x_diff / longer_x)
    return initial_area * (1.0 - earth_area_ratio)



if __name__ == '__main__':
    main()