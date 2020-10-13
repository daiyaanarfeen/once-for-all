import ofa
from ofa.model_zoo import ofa_net
from ofa.tutorial import *

#accuracy_predictor = AccuracyPredictor()
accuracy_predictor = NormPredictor('ofa_mbv3_d234_e346_k357_w1.2_l1_norms.npy')
#efficiency_predictor = LatencyTable()
efficiency_predictor = FLOPsTable()
evolution_finder = EvolutionFinder('flops', 300, efficiency_predictor, accuracy_predictor)

config = evolution_finder.run_evolution_search()
