

import numpy

class Point2D:
    #_x
    #_y
    def __init__(self, x, y):
        self._x = x
        self._y = y

def position1(lC, rC):
    x = (np.abs(rC) - np.abs(lC))/(np.abs(rC) + np.abs(lC))
    y = np.abs(rC - lC)/(np.abs(rC) + np.abs(lC))
    return Point2D(x,y)
    

class TransClassifier:
    # lC, rC : left, right coeffs for same freq
    # _posFunctor( complex lC, rC ) -> [-1, 1]
    # _pMaxX
    # _pMaxY
    def __init__(self, posFunctor):
        self._posFunctor = posFunctor


