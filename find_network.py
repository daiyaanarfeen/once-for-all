import ofa
from ofa.model_zoo import ofa_net
from ofa.tutorial import *

accuracy_predictor = AccuracyPredictor()
#efficiency_predictor = LatencyTable()
efficiency_predictor = FLOPsTable()
evolution_finder = EvolutionFinder('flops', 200, efficiency_predictor, accuracy_predictor)

config = evolution_finder.run_evolution_search()
