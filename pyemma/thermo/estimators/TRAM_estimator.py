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
                 dt_traj='1 step', maxiter=1000, maxerr=1e-5, call_back=None,
                 optimize_lagrangian_mult=False, N_dtram_accelerations=0,
                 dTRAM_mode=False, strict=False, direct_space=False,
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
        self.call_back = call_back
        self._direct_space = direct_space
        self.initialization = initialization
        self.optimize_lagrangian_mult = optimize_lagrangian_mult
        self.N_dtram_accelerations = N_dtram_accelerations
        self._dTRAM_mode = dTRAM_mode
        self._strict = strict
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
        if not self._strict:
            cset = _largest_connected_set(C_sum, directed=True)
        else:
            cset_global = _np.array([],dtype=int)
            for k in range(self.nthermo):
                cset_k = _largest_connected_set(self.count_matrices_full[k,:,:])
                cset_global = _np.union1d(cset_k, cset_global)
            C_conn_strict =  self.count_matrices_full[:, cset_global[:, _np.newaxis], cset_global]
            C_strict_sum = C_conn_strict.sum(axis=0)    
            cset_strict = _largest_connected_set(C_strict_sum)
            cset = cset_global[cset_strict]
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
                if kwargs['iteration'] % 100 == 0:
                     print 'preMBAR', kwargs['iteration_step'], kwargs['error']
            self.mbar_result  = _mbar_direct.estimate(state_counts.sum(axis=1), bias_energy_sequence,
                                               _np.ascontiguousarray(state_sequence[:, 1]),
                                               maxiter=100000, maxerr=1.0E-8)
            therm_energies, self.mbar_unbiased_conf_energies, self.mbar_biased_conf_energies, mbar_error_history = self.mbar_result
            self.biased_conf_energies = self.mbar_biased_conf_energies

            # adapt the Lagrange multiplers to this result
            if self.optimize_lagrangian_mult:
                log_lagrangian_mult = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
                scratch_M = _np.zeros(shape=state_counts.shape[1], dtype=_np.float64)
                _tram.init_lagrangian_mult(self.count_matrices, log_lagrangian_mult)
                new_log_lagrangian_mult = log_lagrangian_mult.copy()
                print 'initializing Lagrange multipliers'
                for _m in range(1000):
                        _tram.update_lagrangian_mult(log_lagrangian_mult, self.mbar_biased_conf_energies, self.count_matrices,
                        state_counts, scratch_M, new_log_lagrangian_mult)
                        nz = _np.where(_np.logical_and(new_log_lagrangian_mult>-30,
                                                       log_lagrangian_mult>-30))
                        if _np.max(_np.abs(new_log_lagrangian_mult[nz] - log_lagrangian_mult[nz])) < self.maxerr:
                            break
                        log_lagrangian_mult[:] = new_log_lagrangian_mult
                self.log_lagrangian_mult = new_log_lagrangian_mult
                print 'done'

        elif self.initialization == 'dTRAM' and self.biased_conf_energies is None:
            occupied = _np.where(state_counts>0)
            def preTRAM_printer(**kwargs):
                if kwargs['iteration'] % 100 == 0:
                     error = _np.max(_np.abs(kwargs['biased_conf_energies'][occupied]-kwargs['old_biased_conf_energies'][occupied]))
                     shape = kwargs['old_biased_conf_energies'].shape
                     argmax = _np.argmax(_np.abs(kwargs['biased_conf_energies']-kwargs['old_biased_conf_energies']))
                     print 'preTRAM', kwargs['iteration'], error, _np.unravel_index(argmax, shape)#, occupied[argmax]
            preTRAM_result = _tram_direct.estimate(_np.zeros_like(self.count_matrices), state_counts, bias_energy_sequence,
                                        _np.ascontiguousarray(state_sequence[:, 1]), maxiter=100000, maxerr=1.E-8, call_back=preTRAM_printer)
            preTRAM_biased_conf_energies = preTRAM_result[0]

            def dTRAM_printer(**kwargs):
                if kwargs['iteration'] % 100 == 0:
                    error = _np.max(_np.abs(kwargs['conf_energies']-kwargs['old_conf_energies']))
                    print 'dTRAM', kwargs['iteration'], error
            dTRAM_biases = preTRAM_biased_conf_energies
            dTRAM_result = _dtram.estimate(self.count_matrices, dTRAM_biases, maxiter=1000000, maxerr=1.E-8, call_back=dTRAM_printer)
            dTRAM_conf_energies = dTRAM_result[1]

            self.log_lagrangian_mult = dTRAM_result[2]
            self.biased_conf_energies = dTRAM_conf_energies + dTRAM_biases

        # run estimator
        if self._dTRAM_mode:
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
                self.count_matrices, state_counts, bias_energy_sequence, _np.ascontiguousarray(state_sequence[:, 1]),
                maxiter = self.maxiter, maxerr = self.maxerr,
                log_lagrangian_mult = self.log_lagrangian_mult,
                biased_conf_energies = self.biased_conf_energies,
                err_out = self.err_out,
                lll_out = self.lll_out,
                callback = self.call_back)

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
