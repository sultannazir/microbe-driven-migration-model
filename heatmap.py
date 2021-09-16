import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

col = ['KE', 'w', 'mH', 'v', 'frac']
data = pd.read_csv('flow_data_job_array.csv', names=col)

KEsplit = []

KEvals = data['KE'].unique()
for i in KEvals:
    KEsplit.append(data[data['KE']==i])

plt.figure(figsize=(20,4.5))
k=1
for i in KEsplit:
    i = i.groupby(['KE','mH','v'],as_index=False).mean()
    dat = i.pivot(index='v', columns='mH', values='frac')
    plt.subplot(1,len(KEvals),k)
    sns.heatmap(dat, vmin=0.25, vmax=0.75)
    plt.title('KE = {}'.format(KEvals[k-1]))
    plt.xlabel('Fraction of host compartment replaced')
    plt.ylabel('Frequency of flow')
    k+=1
plt.tight_layout()
plt.savefig('flow_plot.png')