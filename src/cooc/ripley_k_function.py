'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package. 
@precondition: data.training_file and data.grid_file must be available in configured paths.
@summary: This file includes the functions to analyze the relationship (attraction-repulsion) between bigrams in training data    
'''

import numpy as np
import logging
from scipy import spatial
from geopy.distance import VincentyDistance
from math import sqrt
import random
import geopy
import operator
import cooc
from math import sin, cos, radians
import math
from shapely.geometry import box
from data import datamodel
from _collections import defaultdict

bearing_north = 0.0
bearing_east = 90.0
bearing_south = 180.0
bearing_west = 270.0

def analyze_relationships_of_token(grid, tokens_dictionary, tgms_training, sparse_rowArr, sparse_colArr, token_primary, delta_distance_cartesian, area):
    kScoreAnalyses = []
    tgm_indices_primary = get_tgm_indices_with_token(sparse_rowArr, sparse_colArr, tokens_dictionary[token_primary])
    tgms_primary = [tgms_training[tgm_index] for tgm_index in tgm_indices_primary]
    cooccurrence_counts, cooccurring_tgms_with_tokens = get_ordered_cooccurrences_of_token(token_primary, tgms_primary)
    sorted_cooccurrence_counts = sorted(cooccurrence_counts.items(), key=operator.itemgetter(1), reverse=True)
    for cooccurrence in sorted_cooccurrence_counts:
        if cooccurrence[1] < cooc.MIN_TERM_FREQUENCY:
            break
        tgms_cooccurring = cooccurring_tgms_with_tokens[cooccurrence[0]]
        logging.debug("Analyzing " + token_primary + " (freq:" + str(len(tgms_primary)) + ") and " + cooccurrence[0] + " (freq:" + str(cooccurrence[1]) + ") cooccurring " + str(len(tgms_cooccurring)) + " times") 
        kScoreOfCooccurrence, kScoresOfSimulations = execute_montecarlo(grid, tgms_primary, tgms_cooccurring, delta_distance_cartesian, area)
        kScoreAnalysis = analyze_significance_of_kvalues(token_primary, cooccurrence[0], kScoreOfCooccurrence, kScoresOfSimulations)
        kScoreAnalyses.append(kScoreAnalysis)
    return kScoreAnalyses

def analyze_significance_of_kvalues(token_primary, token_pair, kScoreOfCooccurrence, kScoresOfSimulations):
    kScoresOfSimulations.sort(reverse=True)
    envelope_size = int( cooc.SIGNIFICANCE_RANGE * cooc.N_MONTE_CARLO_SIMULATIONS)
    envelope_upper_boundary = kScoresOfSimulations[envelope_size-1]
    envelope_lower_boundary = kScoresOfSimulations[-envelope_size]
    if kScoreOfCooccurrence < envelope_lower_boundary:
        relationship = cooc.Relationship.REPULSION
    elif kScoreOfCooccurrence > envelope_upper_boundary:
        relationship = cooc.Relationship.ATTRACTION
    else:
        relationship = cooc.Relationship.NOTHING_SIGNIFICANT
    kScoreAnalysis = datamodel.KScoreAnalysis(token_primary, token_pair, relationship, kScoreOfCooccurrence)
    return kScoreAnalysis

def execute_montecarlo(grid, tgms_primary, tgms_cooccurring, delta_distance_cartesian, area):
    points_cartesian_cooccur = [tgm.cartesian_coordinates for tgm in tgms_cooccurring]
    kScoreOfCooccurrence = k_function_cartesian(grid, points_cartesian_cooccur, delta_distance_cartesian, area, cooc.APPLY_EDGE_CORRECTION)
    
    random.seed(len(tgms_primary) * len(tgms_cooccurring))
    points_cartesian_primary = [tgm.cartesian_coordinates for tgm in tgms_primary]
    
    kScoresOfSimulations = []
    for _ in range(cooc.N_MONTE_CARLO_SIMULATIONS):
        latlons_primary3D_random_sample = random.sample(points_cartesian_primary, len(tgms_cooccurring))
        kScoreOfSimulation = k_function_cartesian(grid, latlons_primary3D_random_sample, delta_distance_cartesian, area, cooc.APPLY_EDGE_CORRECTION)
        kScoresOfSimulations.append(kScoreOfSimulation)
    return kScoreOfCooccurrence, kScoresOfSimulations
    
def k_function_cartesian(grid, points_cartesian, delta_distance_cartesian, area, apply_edge_correction):
    number_of_points = len(points_cartesian)
    count_in_range = [0] * number_of_points
    kdTree = spatial.cKDTree(points_cartesian)
    for i in range(number_of_points):
        if apply_edge_correction:
            ratio_of_circle_in_grid = get_ratio_of_circle_in_grid(grid, points_cartesian[i], delta_distance_cartesian)
            edge_correction_multiplier = 1.0 / ratio_of_circle_in_grid
        else:
            edge_correction_multiplier = 1.0
        points_in_range = kdTree.query_ball_point([points_cartesian[i][0], points_cartesian[i][1], points_cartesian[i][2]], r = delta_distance_cartesian)
        count_in_range[i] = edge_correction_multiplier * (len(points_in_range) - 1)
    return float(area * sum(count_in_range)) / (number_of_points**2)

def convert_distance_to_cartesian_at_point(lat, lon, delta_distance_km):
    point_cartesian = latlon_to_cartesian(lat,lon)
    north_lat, north_lon = get_latlon_at_distance(lat, lon, bearing_north, delta_distance_km)
    north_point_cartesian = latlon_to_cartesian(north_lat, north_lon)
    delta_distance_cartesian = sqrt((north_point_cartesian[0] - point_cartesian[0])**2 + (north_point_cartesian[1] - point_cartesian[1])**2 + (north_point_cartesian[2] - point_cartesian[2])**2)
    return delta_distance_cartesian

def get_latlon_at_distance(lat, lon, bearing, distance):
    origin = geopy.Point(lat, lon)
    destination = VincentyDistance(kilometers=distance).destination(origin, bearing)
    lat2, lon2 = destination.latitude, destination.longitude
    return lat2, lon2

def cartesian_to_latlon(x, y, z):
    R = np.float64(6371.000)
    lat = math.degrees(math.asin(z / R))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon

def get_ratio_of_circle_in_grid(grid, circle_latlon3D, distance_range3D):
    grid_latmin, grid_lonmin = grid[0].latmin, grid[0].lonmin
    grid_latmax, grid_lonmax = grid[-1].latmax, grid[-1].lonmax
    grid_sw3D = latlon_to_cartesian(grid_latmin, grid_lonmin)
    grid_ne3D = latlon_to_cartesian(grid_latmax, grid_lonmax)
    if (math.fabs(circle_latlon3D[0] - grid_sw3D[0]) < distance_range3D) or (math.fabs(grid_ne3D[0] - circle_latlon3D[0]) < distance_range3D) or \
        (math.fabs(circle_latlon3D[1] - grid_sw3D[1]) < distance_range3D) or (math.fabs(grid_ne3D[1] - circle_latlon3D[1]) < distance_range3D):
        grid_box = box(grid_sw3D[0], grid_sw3D[1], grid_ne3D[0], grid_ne3D[1])
        circle_region = box(circle_latlon3D[0] - distance_range3D, circle_latlon3D[1] - distance_range3D, circle_latlon3D[0] + distance_range3D, circle_latlon3D[1] + distance_range3D) 
        intersection_area = grid_box.intersection(circle_region).area
        circle_area = circle_region.area
        return intersection_area / circle_area
    return 1.0

def latlon_to_cartesian(lat, lon):
    R = np.float64(6371.000)
    latitude, longitude = map(radians, [lat, lon])
    x = R * cos(latitude) * cos(longitude)
    y = R * cos(latitude) * sin(longitude)
    z = R * sin(latitude)
    return x, y, z

def get_ordered_cooccurrences_of_token(token_primary, tgms_of_token):
    cooccurrence_counts_of_token = defaultdict(int)
    cooccurring_tgms_of_token = defaultdict(list)
    for tgm in tgms_of_token:
        token_indices_primary = [token_index for token_index, token in enumerate(tgm.tokens) if token == token_primary]
        found_token_pairs_in_tgm = set()
        for token_index_primary in token_indices_primary:
            token_index_secondary = token_index_primary + 1
            if token_index_secondary < len(tgm.tokens):
                token_secondary = tgm.tokens[token_index_secondary]
                token_pair = token_primary + cooc.STR_SEPARATOR_FOR_BIGRAMS + token_secondary 
                if token_pair not in found_token_pairs_in_tgm:
                    cooccurrence_counts_of_token[token_pair] += 1
                    cooccurring_tgms_of_token[token_pair].append(tgm)
                    found_token_pairs_in_tgm.add(token_pair)
            token_index_secondary = token_index_primary - 1
            if token_index_secondary >= 0:
                token_secondary = tgm.tokens[token_index_secondary] 
                token_pair = token_secondary + cooc.STR_SEPARATOR_FOR_BIGRAMS + token_primary
                if token_pair not in found_token_pairs_in_tgm:
                    cooccurrence_counts_of_token[token_pair] += 1
                    cooccurring_tgms_of_token[token_pair].append(tgm)
                    found_token_pairs_in_tgm.add(token_pair)
    return cooccurrence_counts_of_token, cooccurring_tgms_of_token

def get_tgm_indices_with_token(sparse_rowArr, sparse_colArr, token_id):    
    row_indexes_with_feature = np.nonzero(sparse_colArr == token_id)
    tgm_indices = sparse_rowArr[row_indexes_with_feature]
    return tgm_indices
