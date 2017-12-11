# Cython version of GIF model with a slow M-current like conductance
# runs with
# Test Mode: cython
# Elapsed time for simulations: all: 0.085814, mean of all except first: 0.000855
# About 23% faster than NUMBA  (weave is deprecated, but also unstable for these runs)
# Numba for identical model takes 0.001107 s for 100 reps (compared to 0.000855 here)
# compile with:
## python setup.py build_ext --inplace
#

import numpy as np
cimport cython
from libc.math cimport exp

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

cdef double alphan(double v):
    return 0.0001*(v + 45)/(1. - exp(-(v + 45.)/9.))

cdef double betan(double v):
    return -0.0001*(v + 45)/(1. - exp((v + 45.)/9.))

cdef double ngate_tau(double v):
    cdef float ntau_inv = (3.0*(alphan(v) + betan(v) ))
    return ntau_inv

cdef double ngate_inf(double v):
    cdef float ntau = 1./(3.0*(alphan(v) + betan(v) ))
    cdef float ninf = alphan(v)*ntau
    return ninf

def ng_tau(v):  # required for python return access... 
    return 1./ngate_tau(v)

def ng_inf(v):
    return ngate_inf(v)

@cython.wraparound(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
def integrate(double[:] T,
            double[:] V,
            double[:] I, 
            double[:] eta_sum, 
            double[:] p_eta, 
            double[:] gamma_sum,
            double[:] p_gamma,
            double[:] p_randnums,
            double[:] spks,
            double[:] nt,
            pars):
        """
        T : np.array (times)
        V : np.array (voltages) 
        I " np.array (current injection)
        eta_sum : np.array for eta pre-spike function
        gamma_sum np.array for gamma post-spike function
        p_rsndnums : a precomputed array of random numbers (set to one value to do dynamically)
        spks : returned array of spike times
        nt : value of n over time (M current activation)
        pars : named tuple of parameters controlling the GIF model
        
        """

        cdef int t, i, j
        cdef float r
        cdef float n_tau, n_inf, n
        cdef float lambdan
        cdef float p_dontspike
        cdef float C = pars.C # pars['C']
        cdef float gl = pars.gl # pars['gl']
        cdef float El  = pars.El # pars['El']
        cdef float gn= pars.gn # pars['gn']
        cdef float En = pars.En # pars['En']
        cdef float Vr = pars.Vr # pars['Vr']
        cdef float dt = pars.dt # pars['dt']
        cdef float lambda0= pars.lambda0 # pars['lambda0']
        cdef int T_ind = pars.T_ind # pars['T_ind']
        cdef int Trefract_ind= pars.Trefract_ind # pars['Trefract_ind']
        cdef float Vt_star = pars.Vt_star # pars['Vt_star']
        cdef float DeltaV = pars.DeltaV # pars['DeltaV'] 
        cdef int p_eta_l = p_eta.shape[0]
        cdef int p_gamma_l = p_gamma.shape[0]
        cdef double vtmp, dtc, inv_dt, inv_deltav
        cdef long int seed = 1
        cdef int npts = 0
        cdef int lastt = 0
        cdef int nrandom = len(p_randnums)
        
        srand48(seed) # initialize random number generator with specified seed
        
#        print('C gl El gn En Vr dt Trefract_ind, Vt_star, DeltaV, V0\n', C, gl, El, gn,
#            En, Vr, dt, Trefract_ind, Vt_star, DeltaV, V[0])
            
        n_tau = ngate_tau(V[0])
        n_inf = ngate_inf(V[0])
        n = n_inf    
        t = 0
        nt[t] = n
        dtc = dt/C
        inv_dt =1./dt
        inv_deltav = 1./DeltaV
        spks = np.zeros(T_ind)
        for i in range(T_ind):
            # INTEGRATE VOLTAGE
            V[t+1] = V[t] + dtc*( - gl*(V[t] - El) - gn*n*(V[t] - En) + I[t] - eta_sum[t] )

           # n_tau = ngate_tau(V[t])  # note result is inverse of tau
            #n_inf = ngate_inf(V[t])
            # advance n
            n = n + (ngate_inf(V[t]) - n)*ngate_tau(V[t])
            nt[t] = n

            vtmp = (V[t+1] - Vt_star - gamma_sum[t]) * inv_deltav
            if vtmp > 700.:  # prevent overflow
                vtmp = 700.
            lambdan = lambda0*exp(vtmp)
            p_dontspike = exp(-lambdan*(dt/1000.0))  # since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)

            # PRODUCE SPIKE STOCHASTICALLY
            #r = np.random.randint(0, 1000)/1000.
            if nrandom < T_ind:
                r = drand48()  # use faster version...
            else:
                r = p_randnums[t]
#            if p_dontspike != 1.0:   # test te values and show how it evilves
#              if t > lastt+2:
#                  print ''
#              print 't %.0f, r: %.4f, p_dontspike: %.4f vtmp: %.4f vt*: %.4f gamma_sum: %.4f, p_gamma_l: %d' % (t, r, p_dontspike, vtmp, Vt_star, gamma_sum[t], p_gamma_l),
#              if (r > p_dontspike):
#                  print  '  ****'
#              else:
#                  print ''
#              lastt = t
                    
            if (r > p_dontspike): # generate a spike  
                if t+1 < T_ind-1:  
                    spks[t+1] = 1.0  # set the spike at t+1
                    nt[t:t+Trefract_ind+1] = nt[t-1]  # save n state as well
                V[t+1:t+Trefract_ind] = 0.  # set v to 0 for duration of spike

                t = t + Trefract_ind   # advance to end of refractory period

                if (t+1) < (T_ind-1):  # are we still in bounds?
                    V[t+1] = Vr  # reset V to Vr
                    nt[t+1] = n  # update state variable
        
                #UPDATE ADAPTATION PROCESSES
                for j in range(p_eta_l):
                    eta_sum[t+1+j] = eta_sum[t+1+j] + p_eta[j]

                for j in range(p_gamma_l):
                    gamma_sum[t+1+j] = gamma_sum[t+1+j] + p_gamma[j]
#                print 'Spike at t= %6d T_ind: %d V: %.2f V+: %.2f gamma_sum: %.4f, r=%.4f  p=%.4f' % (t, T_ind, V[t], V[t+1], gamma_sum[t], r, p_dontspike)    
            t = t + 1
            if t >= T_ind-1:
                break

@cython.wraparound(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
def integrate_forcespikes(double[:] T,
            double[:] V,
            double[:] I, 
            double[:] eta_sum, 
            double[:] p_eta, 
            long[:] spks_i,
            double[:] nt,
                      pars):
        """
        T : np.array (times)
        V : np.array (voltages) 
        I " np.array (current injection)
        eta_sum : np.array for eta pre-spike function
        p_eta : 
        spks_i : indices of spikes to be forced
        nt : value of n over time (M current activation)
        pars : named tuple of parameters controlling the GIF model

        """

        cdef int t, i, j
        cdef float r
        cdef float n_tau, n_inf, n
        cdef float lambdan
        cdef float p_dontspike
        cdef float C = pars.C # pars['C']
        cdef float gl = pars.gl # pars['gl']
        cdef float El  = pars.El # pars['El']
        cdef float gn= pars.gn # pars['gn']
        cdef float En = pars.En # pars['En']
        cdef float Vr = pars.Vr # pars['Vr']
        cdef float dt = pars.dt # pars['dt']
        cdef float lambda0 = pars.lambda0 # pars['lambda0']
        cdef int T_ind = pars.T_ind # pars['T_ind']
        cdef int Trefract_ind = pars.Trefract_ind # pars['Trefract_ind']
        cdef float Vt_star = pars.Vt_star # pars['Vt_star']
        cdef float DeltaV = pars.DeltaV # pars['DeltaV'] 
        cdef float vtmp, dtc, inv_dt, inv_deltav
        cdef long int seed = 1
        cdef int npts = 0
        cdef int spks_cnt = 0
        
        n_tau = ngate_tau(V[0])
        n_inf = ngate_inf(V[0])
        n = n_inf    
        dtc = dt/C


        next_spike = spks_i[0] + Trefract_ind
        spks_cnt = 0
        t = 0
        
        for t in range(T.shape[0]-1):
            # INTEGRATE VOLTAGE
            V[t+1] = V[t] + dtc*( -gl*(V[t] - El) - gn*n*(V[t] - En) + I[t] - eta_sum[t] )

#            n_tau = ngate_tau(vtmp)  # note result is inverse of tau
#            n_inf = ngate_inf(vtmp)
            # advance n
#            n = n + (n_inf - n)*n_tau
#            nt[t] = n

            if t == next_spike:
                spks_cnt = spks_cnt + 1
                next_spike = spks_i[spks_cnt] + Trefract_ind
                V[t-1] = 0                   
                V[t] = Vr
                t = t - 1

            if t >= T.shape[0]-1:
                break
  