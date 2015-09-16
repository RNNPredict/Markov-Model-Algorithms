__author__ = 'wehmeyer, mey'

import numpy as _np
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.util import types as _types
from thermotools import wham as _wham

class WHAM(_Estimator, _MultiThermModel):
    r"""I am a wham class"""
    def __init__(self, b_K_i_full, stride=1, dt_traj='1 step', maxiter=100000, maxerr=1e-5):
        self.b_K_i_full = _types.ensure_ndarray(b_K_i_full, ndim=2, kind='numeric')
        self.stride = stride
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr

    def _estimate(self, trajs):
        """
        Parameters
        ----------
        trajs : ndarray(T, 2) or list of ndarray(T_i, 2)
            Thermodynamic trajectories. Each trajectory is a (T_i, 2)-array
            with T_i time steps. The first column is the thermodynamic state
            index, the second column is the configuration state index.
        """
        # format input if needed
        if isinstance(trajs, _np.ndarray):
            trajs = [trajs]
        # validate input
        assert _types.is_list(trajs)
        for ttraj in trajs:
            _types.assert_array(ttraj, ndim=2, kind='i')
            assert _np.shape(ttraj)[1] == 2
        # harvest state counts
        self.N_K_i_full = _np.zeros((self.nthermo, self.nstates_full, self.nstates_full), dtype=_np.intc)
        for ttraj in trajs:
            for K in range(self.nthermo):
                for i in range(self.nstates_full):
                    self.N_K_i_full[K, i] += (
                        (ttraj[::self.stride, 0] == K) * (ttraj[::self.stride, 1] == i)).sum()
        # active set
        # TODO: check for active thermodynamic set!
        self.active_set = _np.where(self.N_K_i_full.sum(axis=0)> 0)
        self.N_K_i = self.N_K_i_full[:, self.active_set]
        log_N_K = _np.log(self.N_K_i.sum(axis=1)).astype(_np.float64)
        log_N_i = _np.log(self.N_K_i.sum(axis=0)).astype(_np.float64)
        self.b_K_i = _np.ascontiguousarray(self.b_K_i_full[:, self.active_set], dtype=_np.float64)
        # run estimator
        # TODO: use supplied initial guess!
        # TODO: give convergence feedback!
        fi = _np.zeros(shape=log_N_i.shape, dtype=_np.float64)
        fk = _np.zeros(shape=log_N_K.shape, dtype=_np.float64)
        old_fi = _np.empty_like(fi)
        old_fk = _np.empty_like(fk)
        scratch_M = _np.empty_like(fi)
        scratch_T = _np.empty_like(fk)
        for i in range(self.maxiter):
            old_fi[:] = fi[:]
            old_fk[:] = fk[:]
            _wham.iterate_fk(old_fi, self.b_K_i, scratch_M, fk)
            _wham.iterate_fi(log_N_K, log_N_i, fk, self.b_K_i, scratch_T, fi)
            _wham.normalize_fi(fi, scratch_M)
            old_fki = self.b_K_i + old_fi[_np.newaxis, :] - old_fk[:, _np.newaxis]
            fki = self.b_K_i + fi[_np.newaxis, :] - fk[:, _np.newaxis]
            if _np.linalg.norm(old_fki - fki) < self.maxerr:
                break
        # set model parameters to self
        # TODO: find out what that even means...
        self.set_model_params(f_therm=fk, f=fi)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        raise NotImplementedError("I am not implemented")
