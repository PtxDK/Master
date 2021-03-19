from mpunet.augmentation import Elastic2D, Elastic3D


class Elastic2D(Elastic2D):

    def __setstate__(self, state):
        self._alpha = state["alpha"]
        self._sigma = state["sigma"]
        self.apply_prob = state["apply_prob"]
        self.__init__(self._alpha, self._sigma, self.apply_prob)

    def __getstate__(self):
        return self._alpha, self._sigma, self.apply_prob
    
    def __call__(self, x, y):
        return super().__call__(x,y,[0.0]*x.shape[-1])

class Elastic3D(Elastic3D):

    def __setstate__(self, state):
        self._alpha = state["alpha"]
        self._sigma = state["sigma"]
        self.apply_prob = state["apply_prob"]
        self.__init__(self._alpha, self._sigma, self.apply_prob)

    def __getstate__(self):
        return self._alpha, self._sigma, self.apply_prob