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
    from thermotools import mbar as _mbar
    from thermotools import util as _util
except ImportError:
    pass

class TRAM(_Estimator, _MultiThermModel):
    def __init__(self, lag=1, ground_state=None, count_mode='sliding',
                 dt_traj='1 step', maxiter=1000, maxerr=1e-5, err_out=0, lll_out=0):
        self.lag = lag
        self.ground_state = ground_state
        self.count_mode = count_mode
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.err_out = err_out
        self.lll_out = lll_out
        # set cset variable
        self.model_active_set = None
        # set iteration variables
        self.biased_conf_energies = None
        self.log_lagrangian_mult = None

    def _estimate(self, trajs):
        """
        Parameters
        ----------
        trajs : ndarray(X, 2+T) or list of ndarray(X_i, 2+T)
            Thermodynamic trajectories. Each trajectory is a (X_i, 2+T)-array
            with X_i time steps. The first column is the thermodynamic state
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
        cset = _largest_connected_set(C_sum, directed=True)
        self.active_set = cset
        # correct counts
        self.count_matrices = self.count_matrices_full[:, cset[:, _np.newaxis], cset]
        self.count_matrices = _np.require(self.count_matrices, dtype=_np.intc ,requirements=['C', 'A'])
        state_counts = self.state_counts_full[:, cset]
        state_counts = _np.require(state_counts, dtype=_np.intc, requirements=['C', 'A'])
        # create flat bias energy arrays
        state_sequence_full = None
        bias_energy_sequence_full = None
        for traj in trajs:
            if state_sequence_full is None and bias_energy_sequence_full is None:
                state_sequence_full = traj[:, :2]
                bias_energy_sequence_full = traj[:, 2:]
            else:
                state_sequence_full = _np.concatenate(
                    (state_sequence_full, traj[:, :2]), axis=0)
                bias_energy_sequence_full = _np.concatenate(
                    (bias_energy_sequence_full, traj[:, 2:]), axis=0)
        state_sequence_full = _np.ascontiguousarray(state_sequence_full.astype(_np.intc))
        bias_energy_sequence_full = _np.ascontiguousarray(
            bias_energy_sequence_full.astype(_np.float64).transpose())
        state_sequence, bias_energy_sequence = _util.restrict_samples_to_cset(
            state_sequence_full, bias_energy_sequence_full, self.active_set)
        
        # self.test
        assert _np.all(_np.bincount(state_sequence[:, 1]) == state_counts.sum(axis=0))


        # run mbar to generate a good initial guess
        # f_therm, f, self.biased_conf_energies = _mbar.estimate(
        #     self.state_counts.sum(axis=1), bias_energy_sequence,
        #     _np.ascontiguousarray(state_sequence[:, 1]), maxiter=1000, maxerr=1.0E-8)

        # run estimator
        self.biased_conf_energies, conf_energies, therm_energies, self.log_lagrangian_mult, self.err, self.lll = _tram.estimate(
            self.count_matrices, state_counts, bias_energy_sequence, _np.ascontiguousarray(state_sequence[:, 1]),
            maxiter=self.maxiter, maxerr=self.maxerr, err_out=self.err_out, lll_out= self.lll_out,
            log_lagrangian_mult=self.log_lagrangian_mult,
            biased_conf_energies=self.biased_conf_energies)

        # compute models
        scratch_M = _np.zeros(shape=conf_energies.shape, dtype=_np.float64)
        fmsms = [_tram.estimate_transition_matrix(
            self.log_lagrangian_mult, self.biased_conf_energies,
            self.count_matrices, scratch_M, K) for K in range(self.nthermo)]
        self.model_active_set = [_largest_connected_set(msm, directed=False) for msm in fmsms]
        fmsms = [_np.ascontiguousarray(
            (msm[lcc, :])[:, lcc]) for msm, lcc in zip(fmsms, self.model_active_set)]
        models = [_MSM(msm) for msm in fmsms]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=therm_energies, f=conf_energies)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        raise Exception('not implemented')
        #return (self.state_counts * (
        #    self.f_therm[:, _np.newaxis] - self.b_K_i - self.f[_np.newaxis, :])).sum()
