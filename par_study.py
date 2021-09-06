## Study parameters for mean M10 value

import funcs as fn
import numpy as np
import pandas as pd
import csv
import itertools
from joblib import Parallel, delayed
from pathlib import Path

dataName = 'high_freq_high_flow_nomut.csv'
num = 5 # number of replicates per simulation

KErat = np.logspace(-1,1,9)
wvals = np.array([0.1])
reps = np.linspace(1,num, num)

Parameters = {'KH': 50000,  # Carrying capacity in H
              'mu': 0.1,  # microbe mutation rate
              'sim_time': 500,  # total simulation time
              'mH' : 0.5,
              'v' : 0.01,
              'd' : 0.01, # probability of death
              }


def set_parameters(rat, w, r):
    Parameters_local = Parameters.copy()
    Parameters_local['KE'] = int(Parameters['KH'] * rat)
    Parameters_local['w'] = w
    Parameters_local['rep'] = r

    return Parameters_local

def run_model():
    # set modelpar list to run
    modelParList = [set_parameters(*x)
                    for x in itertools.product(*(KErat, wvals, reps))]
    # run model
    nJobs = min(len(modelParList), -1)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(fn.run_one_sim_get_final_state)(par) for par in modelParList)

    # store output
    print(results)

    saveName = dataName
    with open(saveName, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

run_model()

#data = []
#for k in range(len(KErat)):
#    for l in range(len(wvals)):
#        KE = int(Parameters['KH'] * KErat[k])
#        w = wvals[l]
#        Parameters['KE'] = KE
#        Parameters['w'] = w

#        run = fn.run_one_sim_get_final_state(Parameters)
#
#        dat = [KE, w, Parameters['mH'], Parameters['v']] + run
#        data.append(dat)

#        print('KE = {}, w = {} done'.format(KE, w))

#data = pd.DataFrame(data, columns=['KE', 'w', 'NH1', 'NH0', 'NE', 'M01mean', 'M01std', 'M10mean', 'M10std'])

