__author__ = 'wehmeyer, mey'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.thermo import StationaryModel as _StationaryModel
from pyemma.util import types as _types
from thermotools import wham as _wham

class WHAM(_Estimator, _MultiThermModel):
    """
    Example
    -------
    >>> from pyemma.thermo import WHAM
    >>> import numpy as np
    >>> B = np.array([[0, 0],[0.5, 1.0]])
    >>> wham = WHAM(B)
    >>> traj1 = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0]]).T
    >>> traj2 = np.array([[1,1,1,1,1,1,1,1,1,1],[0,1,0,1,0,1,1,0,0,1]]).T
    >>> wham = wham.estimate([traj1, traj2])
    >>> np.around(wham.log_likelihood(), decimals=4)
    -14.1098
    >>> wham.N_K_i
    array([[7, 3],
           [5, 5]], dtype=int32)
    >>> np.around(wham.stationary_distribution, decimals=4)
    array([ 0.5403,  0.4597])
    >>> np.around(wham.meval('stationary_distribution'), decimals=4)
    array([[ 0.5403,  0.4597], [ 0.6597,  0.3403]])
    """
    def __init__(self, b_K_i_full, stride=1, dt_traj='1 step', maxiter=100000, maxerr=1e-5):
        self.b_K_i_full = _types.ensure_ndarray(b_K_i_full, ndim=2, kind='numeric')
        self.stride = stride
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        # set derived quantities
        self.nthermo, self.nstates_full = b_K_i_full.shape

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
        self.N_K_i_full = _np.zeros(shape=(self.nthermo, self.nstates_full), dtype=_np.intc)
        for ttraj in trajs:
            for K in range(self.nthermo):
                for i in range(self.nstates_full):
                    self.N_K_i_full[K, i] += (
                        (ttraj[::self.stride, 0] == K) * (ttraj[::self.stride, 1] == i)).sum()
        # active set
        # TODO: check for active thermodynamic set!
        self.active_set = _np.where(self.N_K_i_full.sum(axis=0) > 0)[0]
        self.N_K_i = _np.ascontiguousarray(self.N_K_i_full[:, self.active_set])
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
        # get stationary models
        sms = [_StationaryModel(
            pi=_np.exp(fk[K, _np.newaxis] - self.b_K_i[K, :] - fi),
            f=self.b_K_i[K, :] + fi - fk[K, _np.newaxis],
            normalize_energy=False, label="K=%d" % K) for K in range(self.nthermo)]
        # set model parameters to self
        # TODO: find out what that even means...
        self.set_model_params(models=sms, f_therm=fk, f=fi)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        return (self.N_K_i * (
            self.f_therm[:, _np.newaxis] - self.b_K_i - self.f[_np.newaxis, :])).sum()
