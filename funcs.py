import numpy as np
import random
import pandas as pd


## Define functions

def parameters_to_global_variables(Parameters):
    ## reads input parameters from dictionary and assigns them as
    ## global variables with variables with names same as the dict keys
    keys = list(Parameters.keys())
    for i in keys:
        globals()[i] = Parameters[i]


def initialize():
    ## 3 arrays, each for each state H0, H1 and E
    ## ech array has 3 rows corresponding to each agent trait:
    ## row 0 = M01
    ## row 1 = M10
    ## row 2 = rH

    ## column index is used to identify a particular agent in the state corresponding to that array.
    ## i.e. H0[0][i] is the M01 value of the i-th agent in state H0

    ## free-floating state in host = empty
    H0 = np.array([[], [], []])

    ## bound state in host = empty
    H1 = np.array([[], [], []])

    ## external environment state
    ## initially all agents are in E, fully adapted to E and bear no mechanisms of attaching to host
    E = np.array([[0] * KE, [1] * KE, [0]*KE])

    return (H0, H1, E)


def flow(H0, E):

    ## initialize a new transition array to temporarily harbour migrating agents
    ## migrants from H0 to E, and their indices:
    migHE = np.array([[], [], []])
    idxHE = []
    ## migrants from E to H0 and their indices:
    migEH = np.array([[], [], []])
    idxEH = []

    NH = len(H0[0])
    NE = len(E[0])

    if NH != 0:
        ## number of emigrants from host
        mHE = int(np.floor(mH * NH))
        ## indices of emigrants
        idxHE = random.sample(range(NH), min(mHE, NH))
        migHE = np.c_[migHE, H0[:, idxHE]]
    if NE != 0:
        ## number of immigrants to host
        mEH = int(np.floor(mH * KH * NE / KE))
        ## indices of emigrants
        idxEH = random.sample(range(NE), min(mEH, NE))
        migEH = np.c_[migEH, E[:, idxEH]]

    ## move migrants from transit to respective states and delete them from initial states
    H0 = np.delete(H0, idxHE, axis=1)
    H0 = np.c_[H0, migEH]
    E = np.delete(E, idxEH, axis=1)
    E = np.c_[E, migHE]

    return (H0, E)


def adhesion(H0, H1):

    ##same method of migration between H0 and E used in flow function but between H1 and H0 now
    mig01 = np.array([[], [], []])
    mig10 = np.array([[], [], []])
    idx01 = []
    idx10 = []
    NH0 = len(H0[0])
    NH1 = len(H1[0])

    ## "migrants" are chosen with probability of attachment and detachment, and not randomly as done in flow function
    if NH0 != 0:
        prob01 = np.random.random(NH0)
        ## agent moves from H0 to H1 w.p. M01
        idx01 = [i for i in range(NH0) if prob01[i] < H0[0][i]]
        mig01 = np.c_[mig01, H0[:, idx01]]
    if NH1 != 0:
        prob10 = np.random.random(NH1)
        ## agent moves from H1 to H0 w.p. M10
        idx10 = [i for i in range(NH1) if prob10[i] < H1[1][i]]
        mig10 = np.c_[mig10, H1[:, idx10]]

    ## move migrants from transit to respective states and delete them from initial states
    H0 = np.delete(H0, idx01, axis=1)
    H0 = np.c_[H0, mig10]
    H1 = np.delete(H1, idx10, axis=1)
    H1 = np.c_[H1, mig01]

    return (H0, H1)

def selection_new(H0, H1, E):
    NH0 = len(H0[0])
    NH1 = len(H1[0])

    if NH0 != 0:
        prob0 = np.random.random(size=(2,NH0))

        ## agents chosen to reproduce with probability rH
        idx0b = [i for i in range(NH0) if prob0[0][i] < H0[2][i]]
        ## add mutations to offsprings within physiological limits
        mutations = np.random.normal(0, mu, size=(3, len(idx0b))).round(2)
        offs = np.clip(H0[:, idx0b] + mutations, 0, 1)

        ## oops some randomly died w.p. d
        idx0d = [i for i in range(NH0) if prob0[1][i] < d]
        H0 = np.delete(H0, idx0d, axis=1)
        H0 = np.c_[H0, offs]

    if NH1 != 0:
        prob1 = np.random.random(size=(2,NH1))

        ## agents chosen to reproduce with probability (1-w)*rH
        idx1b = [i for i in range(NH1) if prob1[0][i] < (1-w)*H1[2][i]]
        ## add mutations to offsprings within physiological limits
        mutations = np.random.normal(0, mu, size=(3, len(idx1b))).round(2)
        offs = np.clip(H1[:, idx1b] + mutations, 0, 1)

        ## oops some randomly died w.p. d
        idx1d = [i for i in range(NH1) if prob1[1][i] < d]
        H1 = np.delete(H1, idx1d, axis=1)
        H1 = np.c_[H1, offs]

    NE = len(E[0])
    if NE != 0:
        prob = np.random.random(size=(2,NE))

        ## agents chosen to reproduce with probability 1-rH
        idxb = [i for i in range(NE) if prob[0][i] < 1 - E[2][i]]
        ## add mutations to offsprings within physiological limits
        mutations = np.random.normal(0, mu, size=(3, len(idxb))).round(2)
        offs = np.clip(E[:, idxb] + mutations, 0, 1)

        ## oops some randomly died w.p. d
        idxd = [i for i in range(NE) if prob[1][i] < d]
        E = np.delete(E, idxd, axis=1)
        E = np.c_[E, offs]

    return(H0, H1, E)

def cap(H0, H1, E):
    NH0 = len(H0[0])
    NH1 = len(H1[0])
    NH = NH0 + NH1
    NE = len(E[0])

    if NH > KH:
        ndel = NH - KH
        ## sample agents from host
        idx = random.sample(range(NH),ndel)
        ## choose agents from state H0
        idx0 = [i for i in idx if i<NH0]
        ## choose agents from state H1
        idx1 = [i-NH0 for i in idx if i>=NH0]
        ## Thanos snapped
        H0 = np.delete(H0, idx0, axis=1)
        H1 = np.delete(H1, idx1, axis=1)

    if NE > KE:
        ndel = NE - KE
        idx = random.sample(range(NE),ndel)
        ## kill the excess chilling in the environments
        E = np.delete(E, idx, axis=1)

    return(H0, H1, E)


## functions for the alder version of the model - can be ignored

def selection_in_host(H0, H1):
    NH0 = len(H0[0])
    NH1 = len(H1[0])
    NH = NH0 + NH1
    p0 = 1 - NH / KH
    p1 = p0 - w

    if NH0 != 0:
        prob0 = np.random.random(NH0)
        if p0 >= 0:
            idx0 = [i for i in range(NH0) if prob0[i] < p0]
            mutations = np.random.normal(0, mu, size=(2, len(idx0))).round(2)
            offs = np.clip(H0[:, idx0] + mutations, 0, 1)
            H0 = np.c_[H0, offs]
        else:
            idx0 = [i for i in range(NH0) if prob0[i] < -p0]
            H0 = np.delete(H0, idx0, axis=1)

    if NH1 != 0:
        prob1 = np.random.random(NH1)
        if p1 >= 0:
            idx1 = [i for i in range(NH1) if prob1[i] < p1]
            mutations = np.random.normal(0, mu, size=(2, len(idx1))).round(2)
            offs = np.clip(H1[:, idx1] + mutations, 0, 1)
            H1 = np.c_[H1, offs]
        else:
            idx1 = [i for i in range(NH1) if prob1[i] < -p1]
            H1 = np.delete(H1, idx1, axis=1)

    return (H0, H1)


def env_dynamics(E):
    NE = len(E[0])
    if NE != 0:
        p = 1 - NE / KE
        prob = np.random.random(NE)
        if p > 0:
            idx = [i for i in range(NE) if prob[i] < p]
            mutations = np.random.normal(0, mu, size=(2, len(idx))).round(2)
            offs = np.clip(E[:, idx] + mutations, 0, 1)
            E = np.c_[E, offs]
        else:
            idx = [i for i in range(NE) if prob[i] < -p]
            E = np.delete(E, idx, axis=1)
    return (E)




## functions to simulate everything - defined for different purposes

def run_one_sim_get_final_state(Parameters):
    # read parameters
    parameters_to_global_variables(Parameters)
    frac = 0

    for rep in range(Parameters['rep']):
        H0, H1, E = initialize()
        for t in range(sim_time):
            vrand = random.random()
            if vrand < v:
                H0, E = flow(H0, E)
            H0, H1 = adhesion(H0, H1)
            H0, H1, E = selection_new(H0, H1, E)
            H0, H1, E = cap(H0, H1, E)

        totH = len(H0[0])+len(H1[0])
        if totH ==0:
            frac += np.nan
        else:
            frac += len(H1[0])/totH

    frac = frac/Parameters['rep']
    data = [Parameters['KE'], Parameters['w'], Parameters['mH'], Parameters['v'], frac]

    print('KE = {}, mH = {}, v = {} done'.format(Parameters['KE'], Parameters['mH'], Parameters['v'] ))
    return (data)

def run_one_sim_get_M(Parameters):
    # read parameters
    parameters_to_global_variables(Parameters)

    H0, H1, E = initialize()
    data = []
    for t in range(sim_time):
        vrand = random.random()
        if vrand < v:
            H0, E = flow(H0, E)
        H0, H1 = adhesion(H0, H1)
        H0, H1 = selection_in_host(H0, H1)
        E = env_dynamics(E)
        NH = len(H0[0]) + len(H1[0])
        if NH == 0:
            data.append([0, 0])
        else:
            data.append([np.mean(np.r_[H0[0], H1[0], E[0]]), np.mean(np.r_[H0[1], H1[1], E[1]])])
    return (data)


def run_one_sim(Parameters):
    # read parameters
    parameters_to_global_variables(Parameters)

    H0, H1, E = initialize()
    data = []
    for t in range(sim_time):
        vrand = random.random()
        if vrand < v:
            H0, E = flow(H0, E)
        H0, H1 = adhesion(H0, H1)
        H0, H1, E = selection_new(H0, H1, E)
        H0, H1, E = cap(H0, H1, E)
        data.append([len(H1[0]), len(H0[0]), len(E[0]), np.histogram(H0[0], bins=20, range=(0, 1))[0],
                     np.histogram(H1[0], bins=20, range=(0, 1))[0],
                     np.histogram(E[0], bins=20, range=(0, 1))[0],
                     np.histogram(H0[1], bins=20, range=(0, 1))[0],
                     np.histogram(H1[1], bins=20, range=(0, 1))[0],
                     np.histogram(E[1], bins=20, range=(0, 1))[0],
                     np.histogram(H0[2], bins=20, range=(0, 1))[0],
                     np.histogram(H1[2], bins=20, range=(0, 1))[0],
                     np.histogram(E[2], bins=20, range=(0, 1))[0]
                     ])
        print(t)
    return (data)

def get_full_data_final(Parameters):
    parameters_to_global_variables(Parameters)
    col = ['M01', 'M10', 'rH', 'State', 'KE']
    H0, H1, E = initialize()
    for t in range(sim_time):
        vrand = random.random()
        if vrand < v:
            H0, E = flow(H0, E)
        H0, H1 = adhesion(H0, H1)
        H0, H1, E = selection_new(H0, H1, E)
        H0, H1, E = cap(H0, H1, E)

        print(t)


    H1dat = pd.DataFrame(np.vstack([H1, np.array(['H1']*len(H1[0])), np.array([Parameters['KE']]*len(H1[0]))]).T,
                         columns=col)
    H0dat = pd.DataFrame(np.vstack([H0, np.array(['H0'] * len(H0[0])), np.array([Parameters['KE']] * len(H0[0]))]).T,
                         columns=col)
    Edat = pd.DataFrame(np.vstack([E, np.array(['E'] * len(E[0])), np.array([Parameters['KE']] * len(E[0]))]).T,
                        columns = col)

    data = pd.concat([H1dat,H0dat,Edat])
    return(data)

