class NoamOpt:
    """ Optimize wrapper that implements rate """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0

    def step(self):
        """ Update parameters and rate """
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        self.optimizer.step()

    def rate(self, step=None):
        """ Implement `lrate` above """
        if step is None:
            step = self._step

        lr = self.factor * \
             (self.model_size ** (-0.5) *
              min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return lr

    def weight_decay(self, step=None):
        # if step is None:
        #     step = self._step
        #
        # if step <= 3000:
        #     weight_decay = 1e-3
        #
        # if 3000 < step <= 9000:
        #     weight_decay = 0.0005
        #
        # if step > 9000:
        #     weight_decay = 1e-4

        weight_decay = 0
        return weight_decay