__author__ = 'wehmeyer, mey, paul'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.msm import MSM as _MSM
from pyemma.util import types as _types
from msmtools.estimation import largest_connected_set as _largest_connected_set
import warnings
import sys

try:
    from thermotools import tram as _tram
    from thermotools import mbar as _mbar
    from thermotools import dtram as _dtram
    from thermotools import util as _util
except ImportError:
    pass

try:
    from thermotools import mbar_direct as _mbar_direct
except ImportError:
    warnings.warn('Direct space implementation of MBAR couldn\'t be imported. TRAM(..., initialization = \'MBAR\', direct_space=True) won\'t work.',  ImportWarning)
try:
    from thermotools import tram_direct as _tram_direct
except ImportError:
    warnings.warn('Direct space implementation of TRAM couldn\'t be imported. TRAM(..., direct_space=True) won\'t work.',  ImportWarning)

class EmptyState(RuntimeWarning):
    pass

class TRAM(_Estimator, _MultiThermModel):
    def __init__(self, lag=1, ground_state=None, count_mode='sliding',
                 dt_traj='1 step', maxiter=1000, maxerr=1e-5, callback=None,
                 N_dtram_accelerations=0,
                 dTRAM_mode=False, direct_space=False,
                 initialization='MBAR', err_out=0, lll_out=0
                 ):
        self.lag = lag
        self.ground_state = ground_state
        self.count_mode = count_mode
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        # set cset variable
        self.model_active_set = None
        # set iteration variables
        self.biased_conf_energies = None
        self.log_lagrangian_mult = None
        self.call_back = callback
        self._direct_space = direct_space
        self.initialization = initialization
        self.N_dtram_accelerations = N_dtram_accelerations
        self._dTRAM_mode = dTRAM_mode
        self.err_out = err_out
        self.lll_out = lll_out

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

        for ttraj in trajs:
            assert ttraj.shape[1] == self.nthermo+2

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
        self.dtrajs_full = []
        for traj in trajs:
            self.dtrajs_full.append(traj[:, 1])
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
        
        # self-test
        assert _np.all(_np.bincount(state_sequence[:, 1]) == state_counts.sum(axis=0))
        assert _np.all(_np.bincount(state_sequence[:, 0]) == state_counts.sum(axis=1))
        assert _np.all(state_counts >= _np.maximum(self.count_matrices.sum(axis=1), self.count_matrices.sum(axis=2)))

        self.state_counts = state_counts

        for k in range(state_counts.shape[0]):
            if state_counts[k,:].sum() == 0:
                warnings.warn('Thermodynamic state %d contains no samples after reducing to the connected set.'%k, EmptyState)
            if self.count_matrices[k,:,:].sum() == 0:
                warnings.warn('Thermodynamic state %d contains no transitions after reducing to the connected set.'%k, EmptyState)

        if self.initialization == 'MBAR' and self.biased_conf_energies is None:
            # initialize with MBAR
            def MBAR_printer(**kwargs):
                if kwargs['iteration_step'] % 100 == 0:
                     print 'preMBAR', kwargs['iteration_step'], kwargs['error']
            self.mbar_result  = _mbar_direct.estimate(state_counts.sum(axis=1), bias_energy_sequence,
                                               _np.ascontiguousarray(state_sequence[:, 1]),
                                               maxiter=100000, maxerr=1.0E-8)
            therm_energies, self.mbar_unbiased_conf_energies, self.mbar_biased_conf_energies, mbar_error_history = self.mbar_result
            self.biased_conf_energies = self.mbar_biased_conf_energies

        # run estimator
        if self._dTRAM_mode: # TODO: remove dTRAM mode
            # use dTRAM (initialized with MBAR) instead of TRAM
            assert self.biased_conf_energies is not None
            print 'Hello dTRAM.'
            dTRAM_biases = self.biased_conf_energies

            def dTRAM_translator(**kwargs):
                if self.call_back is not None:
                    kwargs['biased_conf_energies'] = kwargs['conf_energies'] + dTRAM_biases
                    kwargs['old_biased_conf_energies'] = kwargs['old_conf_energies'] + dTRAM_biases
                    self.call_back(**kwargs)

            dTRAM_result = _dtram.estimate(self.count_matrices, dTRAM_biases, maxiter=1000000, maxerr=1.E-8, call_back=dTRAM_translator)
            dTRAM_conf_energies = dTRAM_result[1]
            self.log_lagrangian_mult = dTRAM_result[2]
            self.biased_conf_energies = dTRAM_conf_energies + dTRAM_biases
            conf_energies = dTRAM_conf_energies
            therm_energies = dTRAM_result[0]
        else:
            if self._direct_space:
                tram = _tram_direct
            else:
                tram = _tram
            self.biased_conf_energies, conf_energies, therm_energies, self.log_lagrangian_mult, self.error_history, self.logL_history = tram.estimate(
                self.count_matrices, self.state_counts, bias_energy_sequence, _np.ascontiguousarray(state_sequence[:, 1]),
                maxiter = self.maxiter, maxerr = self.maxerr,
                log_lagrangian_mult = self.log_lagrangian_mult,
                biased_conf_energies = self.biased_conf_energies,
                err_out = self.err_out,
                lll_out = self.lll_out,
                callback = self.call_back,
                N_dtram_accelerations = self.N_dtram_accelerations)

        # compute and store "mu". TODO: think about storing this directly to a file...
        self.unbiased_pointwise_free_energies = _np.zeros(bias_energy_sequence.shape[1], dtype=_np.float64)
        _tram.get_unbiased_pointwise_free_energies(
            self.log_lagrangian_mult,
            self.biased_conf_energies,
            self.count_matrices,
            bias_energy_sequence,
            _np.ascontiguousarray(state_sequence[:, 1]),
            self.state_counts,
            conf_energies,
            None,    
            None,
            self.unbiased_pointwise_free_energies)

        # compute models
        fmsms = [_tram.estimate_transition_matrix(
            self.log_lagrangian_mult, self.biased_conf_energies,
            self.count_matrices, None, K) for K in range(self.nthermo)]
        self.model_active_set = [_largest_connected_set(msm, directed=False) for msm in fmsms]
        fmsms = [_np.ascontiguousarray(
            (msm[lcc, :])[:, lcc]) for msm, lcc in zip(fmsms, self.model_active_set)]
        models = [_MSM(msm) for msm in fmsms]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=therm_energies, f=conf_energies)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        if self.logL_history is None:
            raise Exception('Computation of log likelihood wasn\'t enabled during estimation.')
        else:
            return self.logL_history[-1]
            
    def pmf(self, x, y, bins):
        # reduce everything to the connected set
        assert len(x)==len(y)==len(self.drajs_full)
        x_sequence = []
        y_sequence = []
        for xtraj, ytraj, traj in zip(x,y,self.drajs_full):
            assert len(xtraj)==len(ytraj)==len(dtraj)
            # TODO: make _util.restrict_samples_to_cset flexible enough to replace the following:
            valid = _np.in1d(dtraj, self.active_set) 
            x_sequence.append(xtraj[valid])
            y_sequence.append(ytraj[valid])
        x_sequence = _np.concatenate(x_sequence)
        y_sequence = _np.concatenate(y_sequence)
        
        # digitize x and y
        x_dsequence = _np.digitize(x_sequence, bins)
        y_dsequence = _np.digitize(y_sequence, bins)
        n = len(bins)+1
        del x_sequence
        del y_sequence
        # generate product indices
        user_index_sequence = y_dsequence * n + x_dsequence
        del x_dsequence
        del y_dsequence

        the_pmf = _np.zeros(shape=n*n, dtype=_np.float64)
        tram.get_unbiased_user_free_energies(
            self.unbiased_pointwise_free_energies,
            user_index_sequence,
            the_pmf)

        return the_pmf.reshape((n,n)), bins, bins # TODO: allow different shapes in x and y
