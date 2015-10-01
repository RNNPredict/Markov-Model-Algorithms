__author__ = 'wehmeyer, mey, paul'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.msm import MSM as _MSM
from pyemma.util import types as _types
from msmtools.estimation import largest_connected_set as _largest_connected_set
from msmtools.estimation import count_matrix as _count_matrix
try:
    from thermotools import tram as _tram
    from thermotools.util import count_matrices as _count_matrices
except ImportError:
    pass

class TRAM(_Estimator, _MultiThermModel):
    def __init__(self, lag=1, ground_state=None, count_mode='sliding', dt_traj='1 step', maxiter=1000, maxerr=1e-5):
        self.lag = lag
        self.ground_state = ground_state
        self.count_mode = count_mode
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        # set derived quantities
        pass
        # set iteration variables
        self.fki = None
        self.log_nuki = None

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
            _types.assert_array(ttraj, ndim=2, kind='f')
            assert _np.shape(ttraj)[1] > 2 # TODO: make strict test
            
        # find dimensions
        self.nstates_full = int(max(_np.max(ttraj[:, 1]) for ttraj in trajs))+1
        self.nthermo = int(max(_np.max(ttraj[:, 0]) for ttraj in trajs))+1
        #print 'M,T:', self.nstates_full, self.nthermo

        # state visitis
        self.N_K_i_full = _np.zeros(shape=(self.nthermo, self.nstates_full), dtype=_np.intc)
        for ttraj in trajs:
            for K in range(self.nthermo):
                for i in range(self.nstates_full):
                    self.N_K_i_full[K, i] += (
                        (ttraj[:, 0] == K) * (ttraj[:, 1] == i)).sum()

        # count matrices
        self.count_matrices_full = _count_matrices(
            [_np.ascontiguousarray(t[:, :2]).astype(_np.intc) for t in trajs], self.lag,
            sliding=self.count_mode, sparse_return=False, nstates=self.nstates_full)

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
        self.fki, fi, fk, self.log_nuki = _tram.estimate(
            self.count_matrices, N_K_i, b_K_x, M_x,
            maxiter=self.maxiter, maxerr=self.maxerr, log_nu_K_i=self.log_nuki, f_K_i=self.fki)

        # get stationary models for the biased ensembles
        scratch = _np.zeros(shape=fi.shape, dtype=_np.float64)
        models = [_MSM(_tram.get_p(self.log_nuki, self.fki, self.count_matrices, scratch, K)) for K in range(self.nthermo)]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=fk, f=fi)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        raise Exception('not implemented')
        #return (self.N_K_i * (
        #    self.f_therm[:, _np.newaxis] - self.b_K_i - self.f[_np.newaxis, :])).sum()
