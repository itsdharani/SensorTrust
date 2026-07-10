import numpy as np

class DisagreementBaseline:
    def __init__(self):
        self.stats = {}
        self.fitted = False

    def fit(self, clean_graph):
        for pair, values in clean_graph.items():
            valid = values[~np.isnan(values)]
            mu = float(np.mean(valid))
            sigma = float(np.std(valid)) + 1e-6
            rectified = np.clip((valid - mu) / sigma, 0, None)
            self.stats[pair] = {
                'mu': mu,
                'sigma': sigma,
                'rect_floor': float(np.mean(rectified))  # <-- the ~0.4 bias, measured not assumed
            }
        self.fitted = True

    def normalize(self, graph):
        if not self.fitted:
            raise RuntimeError("Call fit() on clean data first")
        calibrated = {}
        for pair, values in graph.items():
            mu, sigma, floor = self.stats[pair]['mu'], self.stats[pair]['sigma'], self.stats[pair]['rect_floor']
            rectified = np.clip((values - mu) / sigma, 0, None)
            calibrated[pair] = np.clip(rectified - floor, 0, None)  # subtract the baseline floor
        return calibrated