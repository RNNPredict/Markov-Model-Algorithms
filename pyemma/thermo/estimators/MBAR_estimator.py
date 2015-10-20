__author__ = 'wehmeyer, mey, paul'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.thermo import StationaryModel as _StationaryModel
from pyemma.util import types as _types
from msmtools.estimation import largest_connected_set as _largest_connected_set
from msmtools.estimation import count_matrix as _count_matrix
try:
    from thermotools import mbar as _mbar
except ImportError:
    pass

class MBAR(_Estimator, _MultiThermModel):
    def __init__(self, stride=1, dt_traj='1 step', maxiter=100000, maxerr=1e-5):
        self.stride = stride
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        # set derived quantities
        pass

    def _estimate(self, trajs):
        """
        Parameters
        ----------
        trajs : ndarray(T, 2) or list of ndarray(T_i, 2)
            Thermodynamic trajectories. Each trajectory is a (T_i, 2)-array
            with T_i time steps. The first column is the thermodynamic state
            index, the second column is the configuration state index.
        """
        
        assert self.stride >= 1, "stride < 1 not supported"
        # format input if needed
        if isinstance(trajs, _np.ndarray):
            trajs = [trajs]
        # validate input
        assert _types.is_list(trajs)
        for ttraj in trajs:
            _types.assert_array(ttraj, ndim=2, kind='f')
            assert _np.shape(ttraj)[1] > 2 # TODO: make strict test
            
        # find dimensions
        self.nstates_full = int(max(_np.max(ttraj[:, 1]) for ttraj in trajs))+1
        self.nthermo = int(max(_np.max(ttraj[:, 0]) for ttraj in trajs))+1
        #print 'M,T:', self.nstates_full, self.nthermo
        
        assert self.count_mode=='sliding', 'only sliding window supported at the moment'
        # TODO: computation of N_k_i, M_x, b_K_x without sliding window

        # state visitis
        self.N_K_i_full = _np.zeros(shape=(self.nthermo, self.nstates_full), dtype=_np.intc)
        for ttraj in trajs:
            for K in range(self.nthermo):
                for i in range(self.nstates_full):
                    self.N_K_i_full[K, i] += (
                        (ttraj[:, 0] == K) * (ttraj[:, 1] == i)).sum()

        # count matrix
        self.count_matrices_full = _np.zeros(
            shape=(self.nthermo, self.nstates_full, self.nstates_full), dtype=_np.intc)
        for ttraj in trajs:
            K = int(ttraj[0, 0])
            if not _np.all(ttraj[:, 0] == K):
                raise NotImplementedError("thermodynamic state switching not yet supported")
            self.count_matrices_full[K, :, :] += _count_matrix(
                ttraj[:, 1].astype(_np.intc), self.lag, nstates=self.nstates_full,
                sparse_return=False).astype(_np.intc)

        # restrict to connected set
        C_sum = self.count_matrices_full.sum(axis=0)
        # TODO: report fraction of lost counts
        cset = _largest_connected_set(C_sum)
        self.active_set = cset
        # correct counts
        self.count_matrices = self.count_matrices_full[:,cset[:,_np.newaxis],cset]
        self.count_matrices = _np.require(self.count_matrices, dtype=_np.intc ,requirements=['C','A'])        
        N_K_i = self.N_K_i_full[:,cset]
        N_K_i = _np.require(N_K_i, dtype=_np.intc, requirements=['C','A'])
        # create flat bias energy arrays
        reverse_map = _np.ones(self.nstates_full,dtype=_np.intc)*_np.iinfo(_np.intc).max
        reverse_map[cset] = _np.arange(len(cset))
        M_x = _np.empty(shape=N_K_i.sum(), dtype=_np.intc)
        b_K_x = _np.zeros(shape=(self.nthermo, N_K_i.sum()), dtype=_np.float64)
        i = 0
        for ttraj in trajs:
            valid = _np.where(_np.in1d(ttraj[:, 1], cset))[0]
            M_x[i:i+len(valid)] = reverse_map[ttraj[valid, 1].astype(int)]
            b_K_x[:,i:i+len(valid)] = ttraj[valid, 2:].T
            i += len(valid)
        
        # self.test
        assert _np.all(_np.bincount(M_x) == N_K_i.sum(axis=0))

        # run estimator
        f_K_i, f_i, f_K, log_nu_K_i = _mbar.estimate(
            self.count_matrices, N_K_i, b_K_x, M_x, maxiter=self.maxiter, maxerr=self.maxerr)

        # get stationary models for the biased ensembles
        z_K_i = _np.exp(-f_K_i)
        sms = [_StationaryModel(
            pi=z_K_i[K, :]/z_K_i[K, :].sum(),
            f=f_K_i[K, :],
            normalize_energy=True, label="K=%d" % K) for K in range(self.nthermo)]

        # get stationary model for the unbiased ensemble (bias=0)
        sm0 = _StationaryModel(pi=_np.exp(-f_i), f=f_i, label="unbiased")

        sms = [sm0] + sms

        # set model parameters to self
        self.set_model_params(models=sms, f_therm=f_K, f=f_i)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        raise Exception('not implemented')
        #return (self.N_K_i * (
        #    self.f_therm[:, _np.newaxis] - self.b_K_i - self.f[_np.newaxis, :])).sum()
