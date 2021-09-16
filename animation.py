import funcs as fn
import random
import pandas as pd
import seaborn as sns
from celluloid import Camera
import matplotlib.pyplot as plt
from IPython.display import HTML

Parameters = {'KH': 50000,  # Carrying capacity in H
              'KE' : 500000, # carrying capacity in E
              'mu': 0.1,  # microbe mutation rate
              'sim_time': 50,  # total simulation time
              'v' : 1,   # frequency of flow event
              'mH' : 0.5,  # fraction of host matrix replaced
              'w' : 0.1, # cost of host-bound state
              'd' : 0.01 # probability of intrinsic death
              }

bins = 20
tpf = 400 # milliseconds per time-step/frame
# colour maps for each state
cmap1 = "viridis" # H1
cmap0 = "viridis" # H0
cmap = "viridis"  # E

fig, ax = plt.subplots(figsize=(10, 10), nrows=3, ncols=3)
ax[0][0].set_title('H1')
ax[0][1].set_title('H0')
ax[0][2].set_title('E')

camera = Camera(fig)

fn.parameters_to_global_variables(Parameters)
col = ['M01', 'M10', 'rH']

H0, H1, E = fn.initialize()
for t in range(Parameters['sim_time']):
    vrand = random.random()
    if vrand < Parameters['v']:
        H0, E = fn.flow(H0, E)
    H0, H1 = fn.adhesion(H0, H1)
    H0, H1, E = fn.selection_new(H0, H1, E)
    H0, H1, E = fn.cap(H0, H1, E)

    H1dat = pd.DataFrame(H1.T, columns=col)
    H0dat = pd.DataFrame(H0.T,columns=col)
    Edat = pd.DataFrame(E.T,columns=col)

    sns.histplot(x=H1dat['M01'], y=H1dat['rH'], bins=bins, cmap=cmap1, ax=ax[0][0])
    ax[0][0].set_ylim(0, 1)
    ax[0][0].set_xlim(0, 1)
    sns.histplot(x=H1dat['M10'], y=H1dat['rH'], bins=bins,  cmap=cmap1, ax=ax[1][0])
    ax[1][0].set_ylim(0, 1)
    ax[1][0].set_xlim(0, 1)
    sns.histplot(x=H1dat['M10'], y=H1dat['M01'], bins=bins, cmap=cmap1, ax=ax[2][0])
    ax[2][0].set_ylim(0, 1)
    ax[2][0].set_xlim(0, 1)

    sns.histplot(x=H0dat['M01'], y=H0dat['rH'], bins=bins, cmap=cmap0, ax=ax[0][1])
    ax[0][1].set_ylim(0,1)
    ax[0][1].set_xlim(0, 1)
    sns.histplot(x=H0dat['M10'], y=H0dat['rH'], bins=bins, cmap=cmap0, ax=ax[1][1])
    ax[1][1].set_ylim(0, 1)
    ax[1][1].set_xlim(0, 1)
    sns.histplot(x=H0dat['M10'], y=H0dat['M01'], bins=bins, cmap=cmap0, ax=ax[2][1])
    ax[2][1].set_ylim(0, 1)
    ax[2][1].set_xlim(0, 1)

    sns.histplot(x=Edat['M01'], y=Edat['rH'], bins=bins, cmap=cmap, ax=ax[0][2])
    ax[0][2].set_ylim(0, 1)
    ax[0][2].set_xlim(0, 1)
    sns.histplot(x=Edat['M10'], y=Edat['rH'], bins=bins, cmap=cmap, ax=ax[1][2])
    ax[1][2].set_ylim(0, 1)
    ax[1][2].set_xlim(0, 1)
    sns.histplot(x=Edat['M10'], y=Edat['M01'], bins=bins, cmap=cmap, ax=ax[2][2])
    ax[2][2].set_ylim(0, 1)
    ax[2][2].set_xlim(0, 1)

    plt.tight_layout()
    print(t)

    camera.snap()
animation = camera.animate(interval=400)
HTML(animation.to_html5_video())

plt.show()

#animation.save('sample.mp4')