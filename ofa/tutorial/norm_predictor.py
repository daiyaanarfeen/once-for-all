import numpy as np

class NormPredictor:

    def __init__(self, norm_table):
        self.table = np.load(norm_table)

    def predict_accuracy(self, population):
        k = [3, 5, 7]
        e = [3, 4, 6]
        norms = []
        for sample in population:
            full_depth = [self.table[i, k.index(k_), e.index(e_)] for i, (k_, e_) in enumerate(zip(sample['ks'], sample['e']))]
            stages = [full_depth[4*i: 4*(i+1)] for i in range(5)]
            stages = [s[:d] for (s, d) in zip(stages, sample['d'])]
            norm = sum([sum(s) for s in stages])
            norms.append(norm)

        return np.array(norms)
