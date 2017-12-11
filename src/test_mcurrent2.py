import sys
import numpy as np
import matplotlib.pyplot as mpl
import scipy.stats
from neuron import h
import timeit
from collections import namedtuple
import matplotlib

level = 1
if level == 0:
    from gif.Filter_Rect_LogSpaced import *
    from gif.Filter_Exps import *
    import gif.gif_mcurrent as gif_mcurrent
    import gif.gif_mcurrent_numba
    import gif.gif_mcurrent_weave
    import gif.GIF
    
elif level == 1:
    from Filter_Rect_LogSpaced import *
    from Filter_Exps import *
    import gif_mcurrent as gif_mcurrent
    import gif_mcurrent_numba
    import gif_mcurrent_weave
    import GIF


def print_times(tx, mode):
    print "="*80
    print 'Test Mode: %s' % mode
    print ('Elapsed time for simulations: all: %f, mean of all except first: %f' % (np.sum(tx), np.mean(tx[1:])))
    print ('individual times: ', tx)



def show_sim(res):

    fig, ax = mpl.subplots(5,1)
    ax = ax.ravel()

    mapax = {'weave': 0, 'Numba': 1, 'numba': 1, 'Cython': 2, 'cython': 2, 'square': 2, 'mod': 3}
    for r in res.keys():
        Vs = r[1]
        T = res[r][0]
        for i in range(res[r][1].shape[0]):
            ax[mapax[r]].plot(T, res[r][1][i])

        if len(res[r]) > 2:
            print 'zzz'
            ax[4].plot(T, res[r][2][i])
    mpl.show()


testtypes = ['cython', 'weave', 'numba', 'mod', 'all', 'square']
ntrials = 100

# python setup.py build_ext --inplace
testmode = 'all'  # 'weave' or 'cython' or 'numba'
if len(sys.argv) > 1:
    testmode = sys.argv[1]

    if testmode not in testtypes:
        raise ValueError('Test mode must be in %s or be "all"' % testtypes)
    
if testmode in ['weave', 'all']:
    gif = GIF.GIF()
        
T = np.arange(10000)*0.1


C = 0.2
dt = 0.1
gl = 0.02
El = -60.
gn = 0.02
En = -80.

Vr = -70.
Vt_star = -48.0            # mV, steady state voltage threshold VT*
DeltaV      = 0.5  
Trefract = 1.
Trefract_ind = int(Trefract/dt)
lambda0 = 1.0

eta     = Filter_Rect_LogSpaced()    # nA, spike-triggered current (must be instance of class Filter)
gamma   = Filter_Rect_LogSpaced() 

FEe = Filter_Exps()
FEg = Filter_Exps()

def expfunction_eta(x):
    return 0.2*np.exp(-x/10.0)

eta.setFilter_Function(expfunction_eta)


# Initialize the spike-triggered current gamma with an exponential function        

def expfunction_gamma(x):
    return 10.0*np.exp(-x/100.0)

gamma.setFilter_Function(expfunction_gamma)      

# FEe.setFilter_Timescales([0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 5.0, 10.])
# FEg.setFilter_Timescales([0.3, 0.5, 0.7, 1.0, 1.2, 2.4, 10.0])
# The rest of this test is like GIF.py simulate...
        
T_ind = T.shape[0]

V = np.zeros_like(T)
V[0] = -65.
       
# initialize n

spks_i = np.zeros_like(T)
next_spike = spks_i[0] + Trefract_ind
spks_cnt = 0



(p_eta_support, p_eta) = eta.getInterpolatedFilter(dt)   
p_eta       = p_eta.astype('double')
p_eta_l     = len(p_eta)

(p_gamma_support, p_gamma) = gamma.getInterpolatedFilter(dt)   
p_gamma     = p_gamma.astype('double')
p_gamma_l   = len(p_gamma)

eta_sum = np.zeros(len(T)+2*p_eta_l)
gamma_sum = np.zeros(len(T)+2*p_gamma_l)

spks = np.zeros_like(T)
spks_i = np.zeros_like(T).astype(int)
nt = np.zeros_like(T)

np.random.seed(seed=1)
A = 1
sigma = 10.0
tau=3.0
skew = np.sqrt(2)
#I = 4*(np.random.randn(T.shape[0])+0.5)                                                     
nrand = scipy.stats.skewnorm.rvs(skew, size=int(T.shape[0])) - 0.5
I0 = A * nrand*np.sqrt(2.0*(sigma*sigma)*dt/tau) + 0.2

# for testing, we will use exactly the same random number lists for each method
np.random.seed(456789)
randnums = np.random.randint(0, 10000, size=T.shape[0])/10000.  # 10000 random numbers between 0 and 1


nt = np.zeros_like(T)
t = 0
n = 0
pars  = {
    'Vt_star': Vt_star,
    'Trefract_ind': Trefract_ind,
    'Trefract' : Trefract,
    'T_ind': T_ind,
    'Vr': Vr,
    'DeltaV': DeltaV,
    'C': C,
    'gl': gl,
    'gn': gn,
    'En': En,
    'El': El,
    'dt' : dt,
    'lambda0': lambda0,
    'p_eta_l': len(p_eta),
    'p_gamma_l': len(p_gamma),
    'V0': -65.,
    'seed': 1,  # random number seed.
}

nt_pars = namedtuple('pars', pars.keys())
npars = nt_pars(**pars)  # named tuple for pars for numba
print npars
print p_eta

res = {}

V0 = -65.
npars = nt_pars(**pars)  # named tuple for pars 
if testmode in ['numba', 'all']:  
#    npars = nt_pars(**pars)  # named tuple for pars for numba 
    T = np.arange(10000)*0.1
    V = np.zeros_like(T)
    tx = np.zeros(ntrials)
    Vs = np.zeros((ntrials, T_ind))
#    nt = np.zeros(T_ind)
    ntx = np.zeros((ntrials, T_ind))
    print 'numba'
    randnums = [0]
    I = I0.copy()
    for i in range(ntrials):
        print 'numba: ', i, T_ind, p_eta_l
        eta_sum = np.zeros(T_ind+2*p_eta_l)
        gamma_sum = np.zeros(T_ind+2*p_gamma_l)
        sti = timeit.default_timer()
        gif_mcurrent_numba.integrate(T, V, I, eta_sum, p_eta, gamma_sum, p_gamma, randnums, spks, ntx[i], npars)
        print 'done'
        Vs[i] = V
        tx[i] = timeit.default_timer() - sti

    print_times(tx, 'numba')
    res['numba'] = [T, Vs, ntx]
    
if testmode in ['weave', 'all']:
    npars = nt_pars(**pars)  # named tuple for pars for numba

    tx = np.zeros(ntrials)
    Vs = np.zeros((ntrials, T_ind))
    ntx = np.zeros((ntrials, T_ind))
    gif.setDt(npars.dt)
    T = np.arange(10000)*0.1
    V = np.zeros_like(T)
    I = I0.copy()
    for i in range(ntrials):
        eta_sum = np.zeros(T_ind+2*p_eta_l)
        gamma_sum = np.zeros(T_ind+2*p_gamma_l)
        sti = timeit.default_timer()
        u = gif_mcurrent_weave.integrate(T, V, I, eta_sum, p_eta,
                            gamma_sum, p_gamma, randnums, spks, npars)
        #Vs[i] = v
        (time, Vs[i], eta_sum, V_T, spks) = u
        #(time, Vs[i], eta_sum, V_T, spks) = gif.simulate(I, V0, pars=npars)
        tx[i] = timeit.default_timer() - sti
    print_times(tx, 'weave %d' % i)
    res['weave'] = [time, Vs]
    print 'ok'
    


if testmode in ['cython', 'all']:
    npars = nt_pars(**pars)  # named tuple for pars for numba
    tx = np.zeros(ntrials)
    Vs = np.zeros((ntrials, T_ind))
    ntx = np.zeros((ntrials, T_ind))
    T = np.arange(10000)*0.1
    V = np.zeros_like(T)
    I = I0.copy()
    for i in range(ntrials):
        eta_sum = np.zeros(T_ind+2*p_eta_l)
        gamma_sum = np.zeros(T_ind+2*p_gamma_l)
        sti = timeit.default_timer()
        gif_mcurrent.integrate(T, V, I, eta_sum, p_eta, gamma_sum, p_gamma, randnums, spks, ntx[i], npars)
        Vs[i] = V
        tx[i] = timeit.default_timer() - sti
    print_times(tx, 'cython')
    res['cython'] = [T, Vs, ntx]


if testmode in ['square', 'all']:
    npars = nt_pars(**pars)  # named tuple for pars for numba
    tx = np.zeros(ntrials)
    Vs = np.zeros((ntrials, T_ind))
    ntx = np.zeros((ntrials, T_ind))
    T = np.arange(10000)*0.1
    V = np.zeros_like(T)
    I = np.zeros_like(T)
    for i in range(ntrials):
        eta_sum = np.zeros(T_ind+2*p_eta_l)
        gamma_sum = np.zeros(T_ind+2*p_gamma_l)
        sti = timeit.default_timer()
        IN = I + i*0.005
        gif_mcurrent.integrate(T, V, IN, eta_sum, p_eta, gamma_sum, p_gamma, randnums, spks, ntx[i], npars)
        Vs[i] = V
        tx[i] = timeit.default_timer() - sti
    print_times(tx, 'cython')
    res['cython'] = [T, Vs, ntx]



fig, ax = mpl.subplots(4,1)
ax = ax.ravel()

mapax = {'weave': 0, 'Numba': 1, 'numba': 1, 'Cython': 2, 'cython': 2, 'square': 2}
for r in res.keys():
    print 'r'
    Vs = r[1]
    T = res[r][0]
    for i in range(res[r][1].shape[0]):
        ax[mapax[r]].plot(T, res[r][1][i])

    if len(res[r]) > 2:
        print 'zzz'
        ax[3].plot(T, res[r][2][i])
mpl.show()



