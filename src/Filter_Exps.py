
import numpy as np
from numba import jit

#from scipy.signal import fftconvolve
# from scipy import weave
# from scipy.weave import converters
# import weave
# from weave import converters

import Tools
import sys, traceback
from Filter import *


@jit(nopython=True, cache=True)
def convolve(I, X, R, dt, T_ind, p_taus):

    # CONVOLUTION

    for t in range(T_ind-1):  # (int t=0; t<T_ind-1; t++) {

        for r in range(R): # for (int r=0; r<R; r++)
            X[t+1, r] = (1.0 - dt/p_taus[r])*X[t,r] + I[t]*dt  # X(t+1,r) = (1.0 - dt/p_taus(r))*X(t,r) + I(t)*dt;
    return X

@jit(nopython=True, cache=True)
def convolve_basis(X, R, dt, p_spks_i, p_spks_L, p_T, p_taus):
#    print 'spiketrainbasisfunctions'
    spks_cnt = 0
    next_spike = p_spks_i[0]
    for t in range(p_T-1):
        for r in range(R):
            X[t+1,r] = (1.0- dt/p_taus[r])*X[t,r]        # everybody decay

        if t == next_spike-1:
            for r in range(R):
                X[t+1,r] += 1.0                          # everybody decay and jump
            
            spks_cnt += 1
            if spks_cnt < p_spks_L-1:
                next_spike = int(p_spks_i[spks_cnt])
    return X
    

class Filter_Exps(Filter) :

    """
    Filter f(t) defined as linear combination of exponential functions:
    
    f(t) = sum_j b_j*exp(-t/tau_j),
    
    where b_j is a set of coefficients and tau_j are the time scales associated.
    """

    def __init__(self):
        
        Filter.__init__(self)
         
        # Auxiliary variables that can be computed using the parameters above        
                
        self.taus    = []       # ms, vector defining the timescales {tau_j} used to describe the filter f(t)
 
        self.length_coeff = 10  # the filter length is computed by multiplying the slowest timescale in taus by this coefficient
         
 

    def setFilter_Timescales(self, taus):
 
        """
        Set timescales used to define the filter.
        taus : array values that defines the timescales (in ms)
        """
 
        self.taus = taus
        self.filter_coeffNb = len(taus)
 
     
     
    ############################################################################
    # IMPLEMENT SOME OF THE ABSTRACT METHODS
    ############################################################################
 
    def setFilter_Function(self, f):
        
        print "This function is not yet available."
        
              
    
    def getLength(self):
        
        """
        Return the duration (in ms) of the filter.
        """
            
        return self.length_coeff*max(self.taus)
      
        
    def computeInterpolatedFilter(self, dt):
        
        """
        Compute the interpolated filter self.filter h(t) = sum_j b_j*exp(-t/tau_j) with a given time resolution dt as well as its support self.supportfilter (i.e., time vector)
        """        
        
        # print 'FilterExps: computeInterpolatedFilter'
        # print('self filter_coeffnB %d # self.filter_coeff: %d' % (self.filter_coeffNb, len(self.filter_coeff)))
        # print('filter coeffs:', self.filter_coeff)
        if self.filter_coeffNb != len(self.filter_coeff):
            raise ValueError()
        if self.filter_coeffNb == len(self.filter_coeff) :

            filter_interpol_support = np.arange(self.getLength()/dt)*dt
        
            filter_interpol = np.zeros(len(filter_interpol_support))
        
            for i in np.arange(self.filter_coeffNb) :
              
                filter_interpol += self.filter_coeff[i] * np.exp(-filter_interpol_support/self.taus[i])      
            
                 
            self.filtersupport = filter_interpol_support
            self.filter = filter_interpol
        
        else :
            # traceback.print_exc(file=sys.stdout)
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            # print "*** print_tb:"
            # exc_tracebackp = traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
            # print ("**", p)
            # exit(0)
            raise ValueError("Error: number of filter coefficients does not match the number of basis functions!")
        
        return 0
        

        
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):

        """
        Return matrix containing the result of the convolutional integral between a continuous signal I and all basis functions that define the filter.
        
        I  : numpy array containing the input signal that will be filtered
        dt : timestep used in I
        
        A set of N arrays a1, ..., aN are computed with:
        
        a_i = int_0^t f_i(s)I(t-s)ds
        
        The matrix return by the fuction is made of rows a_i (i.e., the i-th row of the matrix contains a_i)
        """   
        
        # Number of timescales
        R      = int(self.filter_coeffNb)
        
        # Number of time steps
        p_T = int(len(I))
        p_dt = dt
        
        # Timescales
        p_taus = np.array(self.taus)
        p_taus = p_taus.astype("double")

        # Input current
        I = I.astype("double")            
        
        # Matrix in which the result is stored
        # ie, spike train filtered with different basis functions
        X  = np.zeros((p_T,R))
        X  = X.astype("double")

        # for t in range(T_ind):
        #     for r in range(R):
        #         X[t+1, r] = (1.0- p_dt/p_taus(r))*X[t,r] + I[t]*p_dt
        
        X = convolve(I, X, R, dt, p_T, p_taus)
        
        # code moved into numba jit
        # code =  """
        #         #include <math.h>
        #
        #         int   T_ind      = int(p_T);
        #         float dt         = float(p_dt);
        #
        #
        #         // CONVOLUTION
        #
        #         for (int t=0; t<T_ind-1; t++) {
        #
        #             for (int r=0; r<R; r++)
        #                 X(t+1,r) = (1.0 - dt/p_taus(r))*X(t,r) + I(t)*dt;
        #
        #         }
        #
        #         """
        #
        # vars = [ 'p_T', 'p_dt', 'I', 'p_taus', 'X', 'R']
        #
        # v = weave.inline(code, vars, type_converters=converters.blitz)
 
        return X
        
        
    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        """
        Return matrix containing the result of the convolutional integral between a spike train spks and all basis functions that define the filter.
        
        If S(t) is the spike train defined by the spike times in spks, the function should return
        a set of N arrays a1, ..., aN with:
        
        a_i = int_0^t f_i(s)S(t-s)ds
        
        The matrix return by the fuction is made of rows a_i (i.e., the i-th row of the matrix contains a_i)
        """
        
        # Number of timescales
        R      = int(self.filter_coeffNb)
        
        # Number of time steps
        p_T = int(T/dt)
        p_dt = dt
        
        # Timescales
        p_taus = np.array(self.taus)
        p_taus = p_taus.astype("double")

        
        # Spike train
        p_spks_i = Tools.timeToIndex(spks, dt)        # spike times (in time indices)
        p_spks_i = p_spks_i.astype("double")
        p_spks_L = len(p_spks_i)
        
        
        # Matrix in which the result is stored
        # ie, spike train filtered with different basis functions
        X  = np.zeros((p_T,R))
        X  = X.astype("double")

#        print 'spiketrainbasisfunctions'

        X = convolve_basis(X, R, dt, p_spks_i, p_spks_L, p_T, p_taus)
        # code moved to numba jit
        # spks_cnt = 0
        # next_spike = p_spks_i[0]
        # for t in range(p_T-1):
        #     for r in range(R):
        #         X[t+1,r] = (1.0- p_dt/p_taus[r])*X[t,r]        # everybody decay
        #
        #
        #     if t == next_spike-1:
        #
        #         for r in range(R):
        #             X[t+1,r] += 1.0                          # everybody decay and jump
        #
        #         spks_cnt += 1;
        #         if spks_cnt < p_spks_L-1:
        #             next_spike = int(p_spks_i[spks_cnt])



        # original weave code
        # code =  """
        #         #include <math.h>
        #
        #         int   T_ind      = int(p_T);
        #         float dt         = float(p_dt);
        #
        #         //int spks_L     = int(p_spks_L);
        #         int spks_cnt   = 0;
        #         int next_spike = int(p_spks_i(0));
        #
        #
        #         // CONVOLUTION
        #
        #         for (int t=0; t<T_ind-1; t++) {
        #
        #
        #
        #             for (int r=0; r<R; r++)
        #                 X(t+1,r) = (1.0- dt/p_taus(r))*X(t,r);        // everybody decay
        #
        #
        #             if (t == next_spike-1) {
        #
        #                 for (int r=0; r<R; r++) {
        #                     X(t+1,r) += 1.0;                          // everybody decay and jump
        #                 }
        #
        #                 spks_cnt += 1;
        #                 next_spike = int(p_spks_i(spks_cnt));
        #             }
        #
        #         }
        #
        #         """
        #
        # vars = [ 'p_T', 'p_dt', 'p_spks_L', 'p_spks_i', 'p_taus', 'X', 'R']
        #
        # v = weave.inline(code, vars, type_converters=converters.blitz)
      
        return X

