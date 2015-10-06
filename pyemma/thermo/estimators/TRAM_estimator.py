__author__ = 'wehmeyer, mey, paul'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.msm import MSM as _MSM
from pyemma.util import types as _types
from msmtools.estimation import largest_connected_set as _largest_connected_set
try:
    from thermotools import tram as _tram
    from thermotools import util as _util
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
        self.biased_conf_energies = None
        self.log_lagrangian_mult = None

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

        # find state visits and dimensions
        self.state_counts_full = _util.state_counts(trajs)
        self.nstates_full = self.state_counts_full.shape[1]
        self.nthermo = self.state_counts_full.shape[0]

        # count matrices
        self.count_matrices_full = _util.count_matrices(
            [_np.ascontiguousarray(t[:, :2]).astype(_np.intc) for t in trajs], self.lag,
            sliding=self.count_mode, sparse_return=False, nstates=self.nstates_full)

        # restrict to connected set
        C_sum = self.count_matrices_full.sum(axis=0)
        # TODO: report fraction of lost counts
        cset = _largest_connected_set(C_sum)
        self.active_set = cset
        # correct counts
        self.count_matrices = self.count_matrices_full[:, cset[:, _np.newaxis], cset]
        self.count_matrices = _np.require(self.count_matrices, dtype=_np.intc ,requirements=['C', 'A'])        
        state_counts = self.state_counts_full[:, cset]
        state_counts = _np.require(state_counts, dtype=_np.intc, requirements=['C', 'A'])
        # create flat bias energy arrays
        reverse_map = _np.ones(self.nstates_full, dtype=_np.intc) * _np.iinfo(_np.intc).max
        reverse_map[cset] = _np.arange(len(cset))
        state_sequence = _np.empty(shape=state_counts.sum(), dtype=_np.intc)
        bias_energy_sequence = _np.zeros(shape=(self.nthermo, state_counts.sum()), dtype=_np.float64)
        i = 0
        for ttraj in trajs:
            valid = _np.where(_np.in1d(ttraj[:, 1], cset))[0]
            state_sequence[i:i+len(valid)] = reverse_map[ttraj[valid, 1].astype(int)]
            bias_energy_sequence[:,i:i+len(valid)] = ttraj[valid, 2:].T
            i += len(valid)
        
        # self.test
        assert _np.all(_np.bincount(state_sequence) == state_counts.sum(axis=0))

        # run estimator
        self.biased_conf_energies, conf_energies, therm_energies, self.log_lagrangian_mult = _tram.estimate(
            self.count_matrices, state_counts, bias_energy_sequence, state_sequence,
            maxiter=self.maxiter, maxerr=self.maxerr, log_lagrangian_mult=self.log_lagrangian_mult, biased_conf_energies=self.biased_conf_energies)

        # HOTFIX UNTIL MSM SUPPORTS DISCONNECTED SETS:
        from pyemma.thermo import StationaryModel as _StationaryModel
        models = [_StationaryModel(
            pi=_np.exp(therm_energies[K] - self.biased_conf_energies[K, :]), f=self.biased_conf_energies[K, :],
            normalize_energy=False, label="K=%d" % K) for K in range(self.nthermo)]
        #scratch = _np.zeros(shape=conf_energies.shape, dtype=_np.float64)
        #models = [_MSM(_tram.get_p(self.log_lagrangian_mult, self.biased_conf_energies, self.count_matrices, scratch, K)) for K in range(self.nthermo)]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=therm_energies, f=conf_energies)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        raise Exception('not implemented')
        #return (self.state_counts * (
        #    self.f_therm[:, _np.newaxis] - self.b_K_i - self.f[_np.newaxis, :])).sum()
