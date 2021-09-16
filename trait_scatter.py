import funcs as fn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

KErat = [0.1,0.5,2,10]

Parameters = {'KH': 50000,  # Carrying capacity in H
              'mu': 0.1,  # microbe mutation rate
              'sim_time': 500,  # total simulation time
              'v' : 0.8,
              'mH' : 0.2,
              'w' : 0.1, # cost of host-bound state
              'd' : 0.01 # probability of death
              }

fulldata=[]
for K in KErat:

    Parameters['KE'] = int(K*Parameters['KH'])

    data = fn.get_full_data_final(Parameters)
    fulldata.append(data)

df = pd.concat(fulldata)

df.to_csv('medflow.csv')

df = pd.read_csv('medflow.csv')
sns.set_theme(style="darkgrid")

color = 'RdBu'
alpha=1
levels=7
thresh=0.2
H1c = 'Reds'
H0c = 'Blues'

# Set up the figure
f, ax = plt.subplots(figsize=(16, 12), nrows=3, ncols=4)
ax[0][0].set_aspect("equal")

# Draw a contour plot to represent each bivariate density

#### KE=levels000
state = df.query("KE==5000 and State=='H1'")
ax[0][0].set_title('KE=5000')
sns.kdeplot(data=state, x="M01", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][0])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][0])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][0])

state = df.query("KE==5000 and State=='H0'")
sns.kdeplot(data=state, x="M01", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][0])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][0])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][0])

#### KE=25000
state = df.query("KE==25000 and State=='H1'")
ax[0][1].set_title('KE=25000')
sns.kdeplot(data=state, x="M01", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][1])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][1])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][1])

state = df.query("KE==25000 and State=='H0'")
sns.kdeplot(data=state, x="M01", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][1])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][1])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][1])

#### KE=100000
state = df.query("KE==100000 and State=='H1'")
ax[0][2].set_title('KE=100000')
sns.kdeplot(data=state, x="M01", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][2])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][2])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][2])

state = df.query("KE==100000 and State=='H0'")
sns.kdeplot(data=state, x="M01", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][2])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][2])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][2])

#### KE=500000
state = df.query("KE==500000 and State=='H1'")
ax[0][3].set_title('KE=500000')
sns.kdeplot(data=state, x="M01", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][3])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][3])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H1c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][3])

state = df.query("KE==500000 and State=='H0'")
sns.kdeplot(data=state, x="M01", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[0][3])
sns.kdeplot(data=state, x="M10", y="rH", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[1][3])
sns.kdeplot(data=state, x="M10", y="M01", cmap=H0c,
        shade=False, shade_lowest=False, thresh=thresh, levels=levels, alpha=alpha, ax=ax[2][3])

#plt.ylim(0,1)
#plt.xlim(0,1)
plt.tight_layout()
plt.show()