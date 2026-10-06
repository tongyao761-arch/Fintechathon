"""Render fixed iteration diagnostics with bundled matplotlib; no model imports."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
out=Path(args.output)
frame=pd.read_csv(out/'fitting_metrics.csv')
ids=['baseline10','S4R_lean31_minus4','S4R_full34_minus3']
for origin in ['annual_start','midyear']:
    fig,axes=plt.subplots(6,4,figsize=(16,18),layout='constrained')
    for i,name in enumerate(ids):
        for j,year in enumerate([2021,2022,2023,2024]):
            for k,metric in enumerate(['normalized_mse','ic_mean']):
                ax=axes[i*2+k,j]
                sub=frame[(frame.candidate==name)&(frame.year==year)&(frame.origin==origin)]
                for scope,color in [('train','#2070ad'),('valid','#d35b32')]:
                    g=sub[sub.scope==scope].sort_values('iteration')
                    ax.plot(g.iteration,g[metric],marker='o',color=color,label=scope)
                ax.set_title(f'{name.replace("S4R_", "")} / {year}')
                ax.set_xlabel('Trees (fixed prefixes)');ax.set_ylabel(metric)
                ax.set_xticks([20,40,60,80,100]);ax.grid(alpha=.25)
                if k==0:ax.axhline(1,color='gray',linestyle=':',linewidth=.8)
                if i==0 and j==0:ax.legend()
    fig.suptitle(f'{origin}: historical post hoc fitting diagnostics; no tree selection',fontsize=14)
    for suffix in ['png','pdf']:
        dest=out/f'fitting_curves_{origin}.{suffix}'
        if dest.exists():raise FileExistsError(dest)
        fig.savefig(dest,dpi=160)
    plt.close(fig)
