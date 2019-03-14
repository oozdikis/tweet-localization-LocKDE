'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package.
@precondition:
@summary:
'''

STR_SEPARATOR_FOR_BIGRAMS = "|$|"

APPLY_EDGE_CORRECTION = True
SIGNIFICANCE_RANGE = 0.05
N_MONTE_CARLO_SIMULATIONS = 500
KFUNCTION_DELTA_DISTANCE_KM = 0.5
MIN_TERM_FREQUENCY = 5

class Relationship: ATTRACTION, NOTHING_SIGNIFICANT, REPULSION = range(3)

