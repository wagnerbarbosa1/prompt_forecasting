class ExpandingWindow:
    def __init__(self, n_samples, trainw, step, horizon):
        self.n_samples = n_samples
        self.trainw = trainw
        self.step = step
        self.horizon = horizon

    def split(self):
        for k in range(self.trainw, self.n_samples - self.horizon + 1, self.step): #+1 because range delete the superior limit
            #           k é o conjunto de treinamento
            trainidxs = slice(0, k)
            testidxs = slice(k, k + self.horizon)

            yield trainidxs, testidxs

class SlidingWindow:
    def __init__(self, n_samples, trainw, step, horizon):
        self.n_samples = n_samples
        self.trainw = trainw
        self.step = step
        self.horizon = horizon

    def split(self):
        for k in range (self.trainw, self.n_samples - self.horizon + 1, self.step):
            trainidxs = slice(k - self.trainw, k)
            testidxs = slice(k, k + self.horizon)

            yield trainidxs, testidxs