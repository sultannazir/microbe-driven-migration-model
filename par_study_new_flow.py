## Study parameters for mean M10

import funcs as fn
import numpy as np
import csv
import itertools
from joblib import Parallel, delayed

dataName = 'flow_data_new_flow.csv'

svals = [0.1, 0.5, 1, 2, 10]
vvals = [1]
wvals = [0.02, 0.1, 0.5]
mvals = np.linspace(0,1,21)

Parameters = {'KH': 50000,  # Carrying capacity in H
              'Emat': 500000,
              'mu': 0.1,  # microbe mutation rate
              'sim_time': 500,  # total simulation time
              'd' : 0 # probability of death
              }


def set_parameters(w, s, v, mH):
    Parameters_local = Parameters.copy()
    Parameters_local['w'] = w
    Parameters_local['KE'] = int(s*Parameters['Emat'])
    Parameters_local['v'] = v
    Parameters_local['mH'] = mH

    return Parameters_local


def run_model():
    # set modelpar list to run
    modelParList = [set_parameters(*x)
                    for x in itertools.product(*(wvals, svals, vvals, mvals))]
    # run model
    nJobs = min(len(modelParList), -1)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9)(
        [delayed(fn.run_one_sim_get_hist)(par) for par in modelParList])

    output = np.vstack(results)
    # store output

    saveName = dataName
    with open(saveName, "a") as f:
        writer = csv.writer(f)
        writer.writerows(output)

run_model()
