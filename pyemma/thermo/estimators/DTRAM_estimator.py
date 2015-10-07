__author__ = 'noe'

import numpy as _np
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.util import types as _types

class DTRAM(_Estimator, _MultiThermModel):

    def __init__(self, bias_energies_full, lag=1, count_mode='sliding', connectivity='largest',
                 dt_traj='1 step', maxiter=100000, maxerr=1e-5):
        """
        Example
        -------
        >>> from pyemma.thermo import DTRAM
        >>> import numpy as np
        >>> B = np.array([[0, 0],[0.5, 1.0]])
        >>> dtram = DTRAM(B)
        >>> traj1 = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0]]).T
        >>> traj2 = np.array([[1,1,1,1,1,1,1,1,1,1],[0,1,0,1,0,1,1,0,0,1]]).T
        >>> dtram.estimate([traj1, traj2])
        >>> dtram.log_likelihood()
        -9.8058241189353108

        >>> dtram.count_matrices
        array([[[5, 1],
                [1, 2]],

               [[1, 4],
                [3, 1]]], dtype=int32)

        >>> dtram.stationary_distribution
        array([ 0.38173596,  0.61826404])

        >>> dtram.meval('stationary_distribution')
        [array([ 0.38173596,  0.61826404]), array([ 0.50445327,  0.49554673])]

        """
        # set all parameters
        self.bias_energies_full = _types.ensure_ndarray(bias_energies_full, ndim=2, kind='numeric')
        self.lag = lag
        assert count_mode == 'sliding', 'Currently the only implemented count_mode is \'sliding\''
        self.count_mode = count_mode
        assert connectivity == 'largest', 'Currently the only implemented connectivity is \'largest\''
        self.connectivity = connectivity
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr

        # set derived quantities
        self.nthermo, self.nstates_full = bias_energies_full.shape


    def _estimate(self, trajs):
        """
        Parameters
        ----------
        trajs : ndarray(T, 2) or list of ndarray(T_i, 2)
            Thermodynamic trajectories. Each trajectory is a (T, 2)-array
            with T time steps. The first column is the thermodynamic state
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

        # harvest transition counts
        # TODO: replace by an efficient function. See msmtools.estimation.cmatrix
        # TODO: currently dtram_estimator only likes intc, but this should be changed
        self.count_matrices_full = _np.zeros((self.nthermo, self.nstates_full, self.nstates_full), dtype=_np.intc)
        for ttraj in trajs:
            for t in xrange(_np.shape(ttraj)[0]-self.lag):
                self.count_matrices_full[ttraj[t, 0], ttraj[t, 1], ttraj[t+self.lag, 1]] += 1

        # connected set
        from msmtools.estimation import largest_connected_set
        Cs_flat = self.count_matrices_full.sum(axis=0)
        self.active_set = largest_connected_set(Cs_flat)
        self.count_matrices = self.count_matrices_full[:, self.active_set, :][:, :, self.active_set]
        self.count_matrices = _np.ascontiguousarray(self.count_matrices, dtype=_np.intc)
        self.bias_energies = self.bias_energies_full[:, self.active_set]
        self.bias_energies = _np.ascontiguousarray(self.bias_energies, dtype=_np.float64)

        # run estimator
        from pyemma.thermo.pytram.dtram.dtram_estimator import DTRAM as pyemma_DTRAM
        dtram_estimator = pyemma_DTRAM(self.count_matrices, self.bias_energies)
        dtram_estimator.sc_iteration(maxiter=self.maxiter, ftol=self.maxerr, verbose=False)
        self.dtram_estimator = dtram_estimator

        # compute MSM objects
        self.transition_matrices = dtram_estimator.estimate_transition_matrices()
        # TODO: find a robust solution for this (small negative values can occur)
        self.transition_matrices[_np.where(self.transition_matrices < 0)] = 0.0
        for i in xrange(self.nthermo):
            self.transition_matrices[i] /= self.transition_matrices[i].sum(axis=1)[:, None]
        from pyemma.msm import MSM
        msms = [MSM(self.transition_matrices[i, :, :]) for i in xrange(self.nthermo)]
        # compute free energies of thermodynamic states and configuration states
        f_therm = dtram_estimator.f_K
        f_state = dtram_estimator.f_i
        # set model parameters to self
        self.set_model_params(models=msms, f_therm=f_therm, f=f_state)
        # done, return estimator+model
        return self


    def log_likelihood(self):
        nonzero = self.count_matrices.nonzero()
        return _np.sum(self.count_matrices[nonzero] * _np.log(self.transition_matrices[nonzero]))
