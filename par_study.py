## Study parameters for mean M10 value

import funcs as fn
import numpy as np
import csv
import itertools
from joblib import Parallel, delayed

dataName = 'flow_data.csv'

KErat = [0.1,0.5,2,10]
vvals = np.linspace(0.1,1,10)
mvals = np.linspace(0.05,0.5,10)

Parameters = {'KH': 50000,  # Carrying capacity in H
              'mu': 0.1,  # microbe mutation rate
              'sim_time': 500,  # total simulation time
              'w' : 0.1, # cost of host-bound state
              'd' : 0.01 # probability of death
              }


def set_parameters(rat, v, mH):
    Parameters_local = Parameters.copy()
    Parameters_local['KE'] = int(Parameters['KH'] * rat)
    Parameters_local['v'] = v
    Parameters_local['mH'] = mH

    return Parameters_local

def run_model():
    # set model par list to run
    modelParList = [set_parameters(*x)
                    for x in itertools.product(*(KErat, vvals, mvals))]
    # run model
    nJobs = min(len(modelParList), -1)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(fn.run_one_sim_get_final_state)(par) for par in modelParList)

    # store output

    saveName = dataName
    with open(saveName, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

run_model()
