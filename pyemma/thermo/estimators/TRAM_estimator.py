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
    from thermotools import cset as _cset
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
                 connectivity = 'summed_count_matrix', 
                 nn=None, N_dtram_accelerations=0,
                 direct_space=False, report_lost=False,
                 initialization='MBAR', err_out=0, lll_out=0, multi_disc=False
                 ):
        self.lag = lag
        self.ground_state = ground_state
        self.count_mode = count_mode
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.err_out = err_out
        self.lll_out = lll_out
        self.connectivity = connectivity
        self.model_active_set = None
        self.biased_conf_energies = None
        self.mbar_biased_conf_energies = None
        self.log_lagrangian_mult = None
        self.callback = callback
        self.direct_space = direct_space
        self.report_lost = report_lost
        self.initialization = initialization
        self.N_dtram_accelerations = N_dtram_accelerations
        self.err_out = err_out
        self.lll_out = lll_out
        self.nn = nn
        self.multi_disc = multi_disc

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
            assert _np.shape(ttraj)[1] > 2

        # find dimensions
        self.nthermo = int(max(_np.max(ttraj[:, 0]) for ttraj in trajs))+1
        if self.multi_disc:
            self.nstates_full = int(max(_np.max(ttraj[:, 1:1 + self.nthermo]) for ttraj in trajs)) + 1
        else:
            self.nstates_full = int(max(_np.max(ttraj[:, 1]) for ttraj in trajs)) + 1

        for ttraj in trajs:
            if self.multi_disc:
                assert ttraj.shape[1] == 2*self.nthermo + 1
            else:
                assert ttraj.shape[1] == self.nthermo + 2

        # generating_state_trajs contain the origin of every frame
        # (in contrast to conf_state_sequence below where the origin is not encoded)
        if self.multi_disc:
            self.generating_state_trajs_full = []
            for ttraj in trajs:
                 temp = _np.zeros((ttraj.shape[0], 2), dtype=_np.intc)
                 temp[:,0] = ttraj[:, 0].astype(_np.intc)
                 temp[:,1] = ttraj[range(len(ttraj)), 1 + ttraj[:, 0].astype(int)].astype(_np.intc)
                 self.generating_state_trajs_full.append(temp)
                 del temp
        else:
            self.generating_state_trajs_full = [ _np.ascontiguousarray(ttraj[:, 0:2], dtype=_np.intc) for ttraj in trajs ]

        # find state visits and dimensions
        self.state_counts_full = _util.state_counts(self.generating_state_trajs_full)
        self.nstates_full = self.state_counts_full.shape[1]
        self.nthermo = self.state_counts_full.shape[0]

        # count matrices
        self.count_matrices_full = _util.count_matrices(
            self.generating_state_trajs_full, self.lag,
            sliding=self.count_mode, sparse_return=False, nstates=self.nstates_full)

        # restrict to connected set
        tramtrajs_full = _np.concatenate(trajs)
        self.csets, pcset = _cset.compute_csets_TRAM(self.connectivity,
                                                     self.state_counts_full,
                                                     self.count_matrices_full,
                                                     tramtrajs_full,
                                                     nn=self.nn,
                                                     multi_disc=self.multi_disc)
        self.active_set = pcset

        for k in range(self.nthermo):
            if len(self.csets[k]) == 0:
                warnings.warn('Thermodynamic state %d contains no samples after reducing to the connected set.'%k, EmptyState)

        # We don't relabel states anymore, with k-dependent csets that would be too much craziness.
        # Perhaps we should do the conversion of tramtrajs trajectory-wise?
        self.state_counts, self.count_matrices, tramtrajs = _cset.restrict_to_csets(
                                                                self.state_counts_full,
                                                                self.count_matrices_full,
                                                                tramtrajs_full,
                                                                self.csets,
                                                                multi_disc=self.multi_disc)
        if self.report_lost:
            print 'size of projected connected set is', len(pcset)
            N_lost_total = self.state_counts_full.sum() - self.state_counts.sum()
            print 'Totally %d frames were removed' % N_lost_total
            if N_lost_total > 0:
                print 'ensemble, frames lost'
                for k in range(self.nthermo):
                    n_lost_k = self.state_counts_full[k,:].sum() - self.state_counts[k,:].sum()
                    print k, n_lost_k
                print 'conf. state, frames lost'
                for n in range(self.nstates_full):
                    n_lost_n = self.state_counts_full[:,n].sum() - self.state_counts[:,n].sum()
                    print n, n_lost_n

        if self.multi_disc:
            self.conf_state_sequence = tramtrajs[:, 1:1 + self.nthermo].T
            self.bias_energy_sequence = tramtrajs[:, 1 + self.nthermo:].T
        else:
            self.conf_state_sequence = tramtrajs[:, 1]
            self.bias_energy_sequence = tramtrajs[:, 2:].T
        self.conf_state_sequence = _np.require(self.conf_state_sequence, dtype=_np.intc, requirements=['C', 'A'])
        self.bias_energy_sequence = _np.require(self.bias_energy_sequence, dtype=_np.float64, requirements=['C', 'A'])

        # self-test
        assert _np.all(self.state_counts >= _np.maximum(self.count_matrices.sum(axis=1), self.count_matrices.sum(axis=2)))
        assert _np.all(_np.bincount(tramtrajs[:, 0].astype(int), minlength=self.nthermo) == self.state_counts.sum(axis=1))
        if self.multi_disc:
            generating_conf_state_sequence = tramtrajs[range(tramtrajs.shape[0]), 1 + tramtrajs[:, 0].astype(int)].astype(_np.intc)
        else:
            generating_conf_state_sequence = tramtrajs[:, 1].astype(_np.intc)
        assert _np.all(_np.bincount(generating_conf_state_sequence, minlength=self.nstates_full) == self.state_counts.sum(axis=0))
        del generating_conf_state_sequence

        for k in range(self.state_counts.shape[0]):
            if self.count_matrices[k, :, :].sum() == 0:
                warnings.warn('Thermodynamic state %d contains no transitions after reducing to the connected set.'%k, EmptyState)

        if self.initialization == 'MBAR' and self.mbar_biased_conf_energies is None:
            # initialize with MBAR
            def MBAR_printer(**kwargs):
                if kwargs['iteration_step'] % 100 == 0:
                     print 'preMBAR', kwargs['iteration_step'], kwargs['err']

            if self.direct_space:
                mbar = _mbar_direct
            else:
                mbar = _mbar

            if self.multi_disc:
                self.conf_state_sequence_full = tramtrajs_full[:, 1:1 + self.nthermo].T
                self.bias_energy_sequence_full = tramtrajs_full[:, 1 + self.nthermo:].T
            else:
                self.conf_state_sequence_full = tramtrajs_full[:, 1]
                self.bias_energy_sequence_full = tramtrajs_full[:, 2:].T

            self.conf_state_sequence_full = _np.require(self.conf_state_sequence_full, dtype=_np.intc, requirements=['C', 'A'])
            self.bias_energy_sequence_full = _np.require(self.bias_energy_sequence_full, dtype=_np.float64, requirements=['C', 'A'])
            mbar_result  = mbar.estimate(self.state_counts_full.sum(axis=1), self.bias_energy_sequence_full,
                                         self.conf_state_sequence_full,
                                         maxiter=1000000, maxerr=1.0E-8, callback=MBAR_printer, n_conf_states=self.nstates_full)
            self.mbar_therm_energies, self.mbar_unbiased_conf_energies, self.mbar_biased_conf_energies, mbar_error_history = mbar_result
            self.biased_conf_energies = self.mbar_biased_conf_energies.copy()

        # run estimator
        if self.direct_space:
            tram = _tram_direct
        else:
            tram = _tram
        self.biased_conf_energies, conf_energies, self.therm_energies, self.log_lagrangian_mult, self.error_history, self.logL_history = tram.estimate(
            self.count_matrices, self.state_counts, self.bias_energy_sequence, self.conf_state_sequence,
            maxiter = self.maxiter, maxerr = self.maxerr,
            log_lagrangian_mult = self.log_lagrangian_mult,
            biased_conf_energies = self.biased_conf_energies,
            err_out = self.err_out,
            lll_out = self.lll_out,
            callback = self.callback,
            N_dtram_accelerations = self.N_dtram_accelerations)

        # compute models
        scratch_M = _np.zeros(shape=conf_energies.shape, dtype=_np.float64)
        fmsms = [_tram.estimate_transition_matrix(
            self.log_lagrangian_mult, self.biased_conf_energies,
            self.count_matrices, None, K) for K in range(self.nthermo)]
        self.model_active_set = [_largest_connected_set(msm, directed=False) for msm in fmsms]
        fmsms = [_np.ascontiguousarray(
            (msm[lcc, :])[:, lcc]) for msm, lcc in zip(fmsms, self.model_active_set)]
        models = [_MSM(msm) for msm in fmsms]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=self.therm_energies, f=conf_energies)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        if self.logL_history is None:
            raise Exception('Computation of log likelihood wasn\'t enabled during estimation.')
        else:
            return self.logL_history[-1]

    def pointwise_unbiased_free_energies(self, therm_state=None):
        if therm_state is not None:
            assert therm_state<=self.nthermo
        mu_cset = _np.zeros(self.bias_energy_sequence.shape[1], dtype=_np.float64)
        _tram.get_pointwise_unbiased_free_energies(
            therm_state,
            self.log_lagrangian_mult, self.biased_conf_energies,
            self.therm_energies, self.count_matrices,
            self.bias_energy_sequence, self.conf_state_sequence,
            self.state_counts, None, None, mu_cset)
        # Reindex mu such that its indices corresponds to the indices of the
        # dtrajs given by the user (on the full set). Give all samples
        # whose Markov state is not in the connected set a weight of 0.
        j = 0
        mu_trajs = []
        valid = _np.zeros(self.state_counts.shape, dtype=bool)
        for k,cset in enumerate(self.csets):
            valid[k,cset] = True
        for traj in self.generating_state_trajs_full:
            full_size = traj.shape[0]
            ok_traj = valid[traj[:, 0].astype(int), traj[:, 1].astype(int)]
            restricted_size = _np.count_nonzero(ok_traj)
            mu_traj = _np.ones(shape=full_size, dtype=_np.float64)*_np.inf
            mu_traj[ok_traj] = mu_cset[j:j + restricted_size]
            mu_trajs.append(mu_traj)
            j += restricted_size
        assert j==mu_cset.shape[0]
        return mu_trajs

    def mbar_pointwise_unbiased_free_energies(self, therm_state=None):
        if therm_state is not None:
            raise Exception('Choice of therm_state not implemented yet.')
        mu_cset = _np.zeros(self.bias_energy_sequence.shape[1], dtype=_np.float64)
        _mbar.get_pointwise_unbiased_free_energies(
            _np.log(self.state_counts.sum(axis=1)), self.bias_energy_sequence,
            self.mbar_therm_energies, None, mu_cset)
        j = 0
        mu_trajs = []
        valid = _np.zeros(self.state_counts.shape, dtype=bool) # TODO: abstract this a bit
        for k,cset in enumerate(self.csets):
            valid[k,cset] = True
        for traj in self.generating_state_trajs_full:
            full_size = traj.shape[0]
            ok_traj = valid[traj[:,0].astype(int), traj[:,1].astype(int)]
            restricted_size = _np.count_nonzero(ok_traj)
            mu_traj = _np.ones(shape=full_size, dtype=_np.float64)*_np.inf
            mu_traj[ok_traj] = mu_cset[j:j+restricted_size]
            mu_trajs.append(mu_traj)
            j+= restricted_size
        assert j==mu_cset.shape[0]
        return mu_trajs

    # TODO: this is general enough to be used for MBAR as well, move it
    @staticmethod
    def pmf(pointwise_free_energy_trajs, x, y, bins, ybins=None):
        # format input if needed
        if isinstance(pointwise_free_energy_trajs, _np.ndarray):
            pointwise_free_energy_trajs = [pointwise_free_energy_trajs]
        if isinstance(x, _np.ndarray):
            x = [x]
        if isinstance(y, _np.ndarray):
            y = [y]
        # validate input
        assert _types.is_list(pointwise_free_energy_trajs)
        assert _types.is_list(x)
        assert _types.is_list(y)
        assert len(pointwise_free_energy_trajs)==len(x)==len(y)
        for xtraj, ytraj, etraj in zip(pointwise_free_energy_trajs, x, y):
            _types.assert_array(xtraj, ndim=1, kind='f')
            _types.assert_array(ytraj, ndim=1, kind='f')
            _types.assert_array(etraj, ndim=1, kind='f')
            assert len(xtraj)==len(ytraj)==len(etraj)

        if ybins is None:
            ybins = bins
        n = len(bins)+1
        m = len(ybins)+1
        # digitize x and y
        i_sequence = _np.digitize(_np.concatenate(x), bins)
        j_sequence = _np.digitize(_np.concatenate(y), ybins)
        # generate product indices
        user_index_sequence = i_sequence * m + j_sequence
        the_pmf = _np.zeros(shape=n*m, dtype=_np.float64)
        _tram.get_unbiased_user_free_energies(
            _np.concatenate(pointwise_free_energy_trajs),
            user_index_sequence.astype(_np.intc),
            the_pmf)

        return the_pmf.reshape((n,m)), bins, ybins

    @staticmethod
    def expectation(pointwise_free_energies, observable_trajs):
        # TODO: compute per cluster expectations and the global expectation
        # returns (vector of per cluster expectations, global expectation)
        # To to the "per cluster" part, we have to use the conf_state_sequence.
        # Should this be stored in the TRAM object instead of drajs_full?
        pass
