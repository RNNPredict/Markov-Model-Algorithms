__author__='wehmeyer, mey'

import numpy as _np
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.util import types as _types

class WHAM(_Estimator, _MultiThermModel):
    r"""I am a wham class"""
    def __init__(self, b_K_i, stride=1, dt_traj='1 step', maxiter=100000, maxerr=1e-5):
        raise NotImplementedError("I am not implemented")

    def _estimate(self):
        raise NotImplementedError("I am not implemented")

    def log_likelihood(self):
        raise NotImplementedError("I am not implemented")
