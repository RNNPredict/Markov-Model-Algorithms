# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
import unittest
from six.moves import range

import numpy as np
import pyemma.thermo
import msmtools

def tower_sample(distribution):
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, rnd)

def generate_trajectory(transition_matrices, bias_energies, K, n_samples, x0):
    """generates a list of TRAM trajs"""

    traj = np.zeros((n_samples,2+bias_energies.shape[0]))
    x = x0
    traj[0,0] = K
    traj[0,1] = x
    traj[0,2:] = bias_energies[:,x]
    h = 1
    for s in range(n_samples-1):
        x_new = tower_sample(transition_matrices[K,x,:])
        x = x_new
        traj[h,0] = K
        traj[h,1] = x
        traj[h,2:] = bias_energies[:,x]
        h += 1
    return traj

def T_matrix(energy):
    n = energy.shape[0]
    metropolis = energy[np.newaxis, :] - energy[:, np.newaxis]
    metropolis[(metropolis < 0.0)] = 0.0
    selection = np.zeros((n,n))
    selection += np.diag(np.ones(n-1)*0.5,k=1)
    selection += np.diag(np.ones(n-1)*0.5,k=-1)
    selection[0,0] = 0.5
    selection[-1,-1] = 0.5
    metr_hast = selection * np.exp(-metropolis)
    for i in range(metr_hast.shape[0]):
        metr_hast[i, i] = 0.0
        metr_hast[i, i] = 1.0 - metr_hast[i, :].sum()
    return metr_hast


class TestTRAMwith5StateModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bias_energies = np.zeros((2,5))
        cls.bias_energies[0,:] = np.array([1.0,0.0,10.0,0.0,1.0])
        cls.bias_energies[1,:] = np.array([100.0,0.0,0.0,0.0,100.0])
        cls.T = np.zeros((2,5,5))
        cls.T[0,:,:] = T_matrix(cls.bias_energies[0,:])
        cls.T[1,:,:] = T_matrix(cls.bias_energies[1,:])

        n_samples = 50000
        cls.trajs = []
        cls.bias_energies_sh = cls.bias_energies - cls.bias_energies[0,:]
        cls.trajs.append(generate_trajectory(cls.T,cls.bias_energies_sh,0,n_samples,0))
        cls.trajs.append(generate_trajectory(cls.T,cls.bias_energies_sh,0,n_samples,4))
        cls.trajs.append(generate_trajectory(cls.T,cls.bias_energies_sh,1,n_samples,2))
        

    def test_5_state_model(self):
        tram = pyemma.thermo.TRAM(lag=1, maxerr=1E-13, lll_out=10, direct_space=False)
        tram.estimate(self.trajs)

        log_pi_K_i = tram.biased_conf_energies.copy()
        log_pi_K_i[0,:] -= np.min(log_pi_K_i[0,:])
        log_pi_K_i[1,:] -= np.min(log_pi_K_i[1,:])

        assert np.allclose(log_pi_K_i, self.bias_energies, atol=0.1)

        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(tram.logL_history[-1]+1.E-5>=tram.logL_history[0:-1])

        # simple test: just call the methods
        mu = tram.pointwise_unbiased_free_energies()
        x = [traj[:,1] for traj in self.trajs]
        pyemma.thermo.TRAM(mu, x, x, np.arange(0,4).astype(np.float64))

    def test_5_state_model_direct(self):
        tram = pyemma.thermo.TRAM(lag=1, maxerr=1E-13, lll_out=10, direct_space=True)
        tram.estimate(self.trajs)

        log_pi_K_i = tram.biased_conf_energies.copy()
        log_pi_K_i[0,:] -= np.min(log_pi_K_i[0,:])
        log_pi_K_i[1,:] -= np.min(log_pi_K_i[1,:])

        assert np.allclose(log_pi_K_i, self.bias_energies, atol=0.1)

        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(tram.logL_history[-1]+1.E-5>=tram.logL_history[0:-1])

class TestTRAMasReversibleMSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_states = 50
        traj_length = 10000

        traj = np.zeros(traj_length, dtype=int)
        traj[::2] = np.random.randint(1,n_states,size=(traj_length-1)//2+1)

        c = msmtools.estimation.count_matrix(traj, lag=1)
        while not msmtools.estimation.is_connected(c, directed=True):
            traj = np.zeros(traj_length, dtype=int)
            traj[::2] = np.random.randint(1,n_states,size=(traj_length-1)//2+1)
            c = msmtools.estimation.count_matrix(traj, lag=1)

        state_counts = np.bincount(traj)[:,np.newaxis]
        cls.tram_traj = np.zeros((traj_length,3))
        cls.tram_traj[:,1] = traj

        cls.T_ref = msmtools.estimation.tmatrix(c, reversible=True).toarray()

    def test_reversible_msm(self):
        tram = pyemma.thermo.TRAM(lag=1,maxerr=1.E-20, lll_out=10, direct_space=False)
        tram.estimate(self.tram_traj)
        #pos = np.unravel_index(np.argmax(np.abs(T_ref-tram.models[0].transition_matrix)),T_ref.shape)
        assert np.allclose(self.T_ref,  tram.models[0].transition_matrix, atol=1.E-4)

        # Lagrange multipliers should be > 0
        assert np.all(tram.log_lagrangian_mult > -1.E300)
        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(tram.logL_history[-1]+1.E-5>=tram.logL_history[0:-1])

    def test_reversible_msm_direct(self):
        tram = pyemma.thermo.TRAM(lag=1,maxerr=1.E-20, lll_out=10, direct_space=True)
        tram.estimate(self.tram_traj)
        #pos = np.unravel_index(np.argmax(np.abs(T_ref-tram.models[0].transition_matrix)),T_ref.shape)
        assert np.allclose(self.T_ref,  tram.models[0].transition_matrix, atol=1.E-4)

        # Lagrange multipliers should be > 0
        assert np.all(tram.log_lagrangian_mult > -1.E300)
        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(tram.logL_history[-1]+1.E-5>=tram.logL_history[0:-1])


if __name__ == "__main__":
    unittest.main()


