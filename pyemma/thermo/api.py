
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__author__ = 'noe'

import numpy as _np
from pyemma.util import types as _types

# ===================================
# Data Loaders and Readers
# ===================================

# TODO: what about simple molecular dynamics data? How do we combine MD data with US data?

# This corresponds to the source function in coordinates.api
def umbrella_sampling_data(umbrella_trajs, centers, k, md_trajs = None, nbin=None):
    r""" Wraps umbrella sampling data or a mix of umbrella sampling and and direct molecular dynamics

    Parameters
    ----------
    umbrella_trajs : list of K arrays, each of shape (T_i, d)
        List of arrays, each having T_i rows, one for each time step, and d columns where d is the dimension in which
        umbrella sampling was applied. Often d=1, and thus umbrella_trajs will be a list of 1d-arrays.
    centers : array-like of size K
        list or array of K center positions. Each position must be a d-dimensional vector. For 1d umbrella sampling,
        one can simply pass a list of centers, e.g. [-5.0, -4.0, -3.0, ... ]
    k : int or array-like of int
        the force constant used in the umbrellas, unit-less (e.g. kT per length unit). If different force constants
        were used for different umbrellas, a list or array of K force constants can be given.
        For multidimensional umbrella sampling, the force matrix must be used.
    mdtrajs : list of K arrays, each of shape (T_i, d), optional, default = None
        unbiased molecular dynamics simulations. Format like umbrella_trajs.
    nbin

    """
    pass

# This corresponds to the source function in coordinates.api
def multitemperature_to_bias(etrajs, ttrajs, kTs):
    r""" Wraps umbrella sampling data or a mix of umbrella sampling and and direct molecular dynamics

    The probability at the thermodynamic ground state is:

    .. math:
        \pi(x) = \mathrm{e}^{-\frac{U(x)}{kT_{0}}}

    The probability at excited thermodynamic states is:

    .. math:
        \pi^I(x) = \mathrm{e}^{-\frac{U(x)}{kT_{I}}}
                 = \mathrm{e}^{-\frac{U(x)}{kT_{0}}+\frac{U(x)}{kT_{0}}-\frac{U(x)}{kT_{I}}}
                 = \mathrm{e}^{-\frac{U(x)}{kT_{0}}}\mathrm{e}^{-\left(\frac{U(x)}{kT_{I}}-\frac{U(x)}{kT_{0}}\right)}
                 = \mathrm{e}^{-u(x)}\mathrm{e}^{-b_{I}(x)}

    where we have defined the bias energies:

    .. math:
        b_{I}(x) = U(x)\left(\frac{1}{kT_{I}}-\frac{1}{kT_{0}}\right)


    Parameters
    ----------
    etrajs : ndarray or list of ndarray
        energy trajectories
    kTs : ndarray of float
        kT values of the different temperatures

    """
    pass

# ===================================
# Estimators
# ===================================

def tram(dtrajs, ttrajs, btrajs, lag, ground_state=0, maxiter=100000, maxerr=1.0E-5):
    # TODO: describe ftol precisely
    """Transition-based reweighting analysis method

    Estimates a multi-thermodynamic Markov state model (MT-MSM) including an optimal estimate of the thermodynamics
    and kinetics at multiple thermodynamic states. The input are simulation data from multiple thermodynamic states,
    e.g. direct molecular dynamics at different temperatures, umbrella sampling, parallel tempering, metadynamics,
    or any mixture of them.

    Parameters
    ----------
    dtrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes in 1,...,n
        enumerating the n Markov states or the bins the trajectory is in at any time.
    ttrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes in 1,...,K
        enumerating the thermodynamic states the trajectory is in at any time.
    btrajs : ndarray(T,K) of float, or list of ndarray(T_i,K) of float
        A single trajectory or a list of trajectories of bias energies. Each frame has K bias energies, evaluating
        the energy of the current configuration in each of the K thermodynamic states.
    ground_state : int, optional, default=0
        index of the thermodynamic ground or reference state. The statistics of
        all thermodynamic states will be computed, but the state probabilities
        of this state will be set to the default state probabilities in the
        TRAM object that is returned
    maxiter : int, optional, default=100000
        The maximum number of TRAM iterations before the estimator exits unsuccessfully
    ftol : float, optional, default=1e-5
        Convergence criterion based on the max change in an self-consistent-iteration step

    Returns
    -------
    mtmsm : MultiThermMSM
        A multi-thermodynamic Markov state model which consists of stationary and kinetic quantities at all
        temperatures.

    Example
    -------
    **Example 1: Umbrella sampling**. Suppose we simulate in K umbrellas, centered at positions :math:`y_1,...,y_K`
    with bias energies

    .. math::
        b_k(x) = c (x - y_k)^2

    Suppose we have one simulation of length T in each umbrella, and they are ordered from 1 to K. We have discretized
    the x-coordinate into 100 bins.

    Then dtrajs, ttrajs and btrajs should each be a list of :math:`K` arrays.
    dtrajs would look for example like this:

    [ (1, 2, 2, 3, 2, ...),  (2, 4, 5, 4, 4, ...), ... ]

    where each array has length T, and is the sequence of bins (in the range 0 to 99) visited along the trajectory.
    ttrajs would look like this:

    [ (0, 0, 0, 0, 0, ...),  (1, 1, 1, 1, 1, ...), ... ]

    Because trajectory 1 stays in umbrella 1 (index 0), trajectory 2 stays in umbrella 2 (index 1), and so forth.
    btrajs is a list of K arrays, each having size T x K, e.g.:

    [[b_0(x0), b_1(x0), ..., b_K(x0)],
     [b_0(x1), b_1(x1), ..., b_K(x1)],
     ...
     [b_0(xT), b_1(xT), ..., b_K(xT)]]

    where xt is the x-coordinate in trajectory frame t, and the energy computed by the bias equation above.

    **Example 2: Parallel tempering**. Suppose we simulate at K temperatures and consider a switching between
    temperatures at time intervals corresponding to two saved time steps. We can choose to arrange trajectories
    either by replica or by temperature. If we give them by replica, the dtrajs might look like this:

    [ (1, 2, 9, 8, 1, ...),  (7, 8, 2, 2, 7, ...), ... ]

    and the ttrajs would look like:

    [ (0, 0, 1, 1, 0, ...),  (1, 1, 0, 0, 1, ...), ... ]

    So you see that at time step 3 and 5 we have exchanged replicas at the lowest and second-lowest temperatures.
    If the total length of our simulation is T time steps, btrajs will be a list of K arrays of size T x K. The
    first btraj would look like:

    [[b_0,0(x0), b_0,1(x0), ..., b_0,K(x0)],
     [b_0,0(x1), b_0,1(x1), ..., b_0,K(x1)],
     [b_1,0(x1), b_1,1(x1), ..., b_1,K(x1)],
     [b_1,0(x1), b_1,1(x1), ..., b_1,K(x1)],
     [b_0,0(x1), b_0,1(x1), ..., b_0,K(x1)],
     ...
    ]

    where

    .. math::
        b_i,j(x) = \frac{U(x)}{k_B} \left( \frac{1}{T_j} - \frac{1}{T_i} \right)

    is the bias energy involved in reweighting configuration x from the generating temperature :math:`T_i` to another
    temperature :math:`T_j`

    """
    # prepare trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    btrajs = _types.ensure_traj_list(btrajs)
    ntrajs = len(ttrajs)
    assert len(dtrajs) == ntrajs
    assert len(btrajs) == ntrajs
    nthermo = btrajs[0].shape[1]
    X = []
    for i in xrange(ntrajs):
        assert len(dtrajs[i]) == len(ttrajs[i])
        assert len(btrajs[i]) == len(ttrajs[i])
        X.append(_np.hstack([ttrajs[i][:, None], dtrajs[i][:, None], btrajs[i]]))
    # build XTRAM
    from pyemma.thermo.estimators import TRAM
    xtram_estimator = TRAM(lag=lag, ground_state=0, count_mode='sliding', maxiter=maxiter, maxerr=maxerr)
    # run estimation
    return xtram_estimator.estimate(X)


# TODO: B is transposed compared to the tram array. Should we transpose B for consistency?
def dtram(ttrajs, dtrajs, B, lag, maxiter=100000, maxerr=1.0E-5):
    """Discrete transition-based reweighting analysis method

    Parameters
    ----------
    ttrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes in 1,...,K
        enumerating the thermodynamic states the trajectory is in at any time.
    dtrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes in 1,...,n
        enumerating the n Markov states or the bins the trajectory is in at any time.
    B : ndarray(K, n)
        b[j,i] is the bias energy for each discrete state i at thermodynamic state j.

    See also
    --------
    tram

    """
    # prepare trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    assert len(ttrajs) == len(dtrajs)
    X = []
    for i in xrange(len(ttrajs)):
        ttraj = ttrajs[i]
        dtraj = dtrajs[i]
        assert len(ttraj) == len(dtraj)
        X.append(_np.array([ttraj, dtraj]).T)
    # build DTRAM
    from pyemma.thermo.estimators import DTRAM
    dtram_estimator = DTRAM(bias_energies_full=B, lag=lag, count_mode='sliding', maxiter=maxiter, maxerr=maxerr)
    # run estimation
    return dtram_estimator.estimate(X)

