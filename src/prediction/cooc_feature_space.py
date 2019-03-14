'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package.
@precondition:
@summary: This file includes the functions to extend the feature space with bigrams.
'''

import logging
from _collections import defaultdict
import cooc
import data
from data import datareader

def replace_bigrams_in_tgms(tgms, replace_token_pairs):
    new_tokens = set()
    for tgm in tgms:
        if len(tgm.tokens) < 2:
            continue
        combined_tokens_to_add = set()
        token_indices_to_remove = set()
        for i in range(len(tgm.tokens)):
            token_primary = tgm.tokens[i]
            if i < len(tgm.tokens) - 1:
                token_secondary = tgm.tokens[i+1]
                combined_token = token_primary + cooc.STR_SEPARATOR_FOR_BIGRAMS + token_secondary
                if replace_token_pairs.has_key(token_primary) and combined_token in replace_token_pairs[token_primary]:
                    combined_tokens_to_add.add(combined_token)
                    new_tokens.add(combined_token)
                    token_indices_to_remove.add(i)
            if i > 0:
                token_secondary = tgm.tokens[i-1]
                combined_token = token_secondary + cooc.STR_SEPARATOR_FOR_BIGRAMS + token_primary
                if replace_token_pairs.has_key(token_primary) and combined_token in replace_token_pairs[token_primary]:
                    combined_tokens_to_add.add(combined_token)
                    new_tokens.add(combined_token)
                    token_indices_to_remove.add(i)
        if len(token_indices_to_remove) > 0:
            token_indices_to_remove = list(token_indices_to_remove)
            token_indices_to_remove.sort(reverse=True)
            for i in token_indices_to_remove:
                tgm.tokens.pop(i)
        if len(combined_tokens_to_add) > 0:
            for combined_token_to_add in combined_tokens_to_add:
                tgm.tokens.append(combined_token_to_add)
    logging.debug("Number of new_tokens added to the dictionary: " + str(len(new_tokens)))
    return new_tokens

def find_unigrams_that_no_longer_exist(tgms_training, selected_tokens_set):
    all_tokens_in_tgms_training = set()
    for tgm in tgms_training:
        for token in tgm.tokens:
            all_tokens_in_tgms_training.add(token)
    unigrams_that_no_longer_exist = selected_tokens_set.difference(all_tokens_in_tgms_training)
    logging.debug("unigrams_that_no_longer_exist: " + str(len(unigrams_that_no_longer_exist)))
    return unigrams_that_no_longer_exist
    

def generate_bigrams_from_unigrams(tgms, bigrams_set):
    for tgm in tgms:
        if len(tgm.tokens) < 2:
            tgm.tokens = []
            continue
        bigrams_to_add = set()
        for i in range(len(tgm.tokens) - 1):
            token_primary = tgm.tokens[i]
            token_secondary = tgm.tokens[i+1]
            bigram = token_primary + cooc.STR_SEPARATOR_FOR_BIGRAMS + token_secondary
            if bigram in bigrams_set:
                bigrams_to_add.add(bigram)
        tgm.tokens = []
        for bigram in bigrams_to_add:
            tgm.tokens.append(bigram)

def merge_token_pairs(token_pairs1, token_pairs2):
    merged_token_pairs = defaultdict(set)
    keys = token_pairs1.keys()
    for key in keys:
        merged_token_pairs[key] = set(token_pairs1[key])
    keys = token_pairs2.keys()
    for key in keys:
        if merged_token_pairs.has_key(key):
            merged_token_pairs[key].update(token_pairs2[key])
        else:
            merged_token_pairs[key] = set(token_pairs2[key])
    return merged_token_pairs

def extend_feature_space_with_bigrams(tgms_training, tgms_test, tokens_set):
    attraction_token_pairs, repulsion_token_pairs = datareader.read_kscore_token_pairs(data.kscore_analysis_file) 
    if len(attraction_token_pairs) == 0 and len(repulsion_token_pairs) == 0:
        logging.warn('There are no attraction or repulsion token pairs')
        return
    logging.debug("Found " + str(sum(len(v) for v in attraction_token_pairs.itervalues())) + " attraction_token_pairs and " + str(sum(len(v) for v in repulsion_token_pairs.itervalues())) + " repulsion_token_pairs")
    merged_token_pairs = merge_token_pairs(attraction_token_pairs, repulsion_token_pairs)
    new_bigrams_in_training_tgms = replace_bigrams_in_tgms(tgms_training, merged_token_pairs)
    replace_bigrams_in_tgms(tgms_test, merged_token_pairs)

    unigrams_to_delete = find_unigrams_that_no_longer_exist(tgms_training, tokens_set)
    for bigram in new_bigrams_in_training_tgms:
        tokens_set.add(bigram)
    for unigram in unigrams_to_delete:
        tokens_set.remove(unigram)



