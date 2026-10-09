"""Plot saved aggregate outputs only. No models or hypothesis tests are run.
Run: python plot_statewise.py --input statewise_primary_results.csv --output .
"""
from pathlib import Path
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('.'))
    a = p.parse_args()
    r = pd.read_csv(a.input)
    r = r[r.metric.eq('observed_gap_pp')].copy()
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    colors = {'NFHS-4':'#17678A','NFHS-5':'#A74D2C'}
    for cohort, heading in [('all_narrow','All eligible narrow births'),('first_delivery','First-delivery births')]:
        fig, axes = plt.subplots(1,2,figsize=(16,14))
        limits = [-40,100]
        for ax,wave in zip(axes,['NFHS-4','NFHS-5']):
            sub=r[r.wave.eq(wave)&r.analysis.eq(cohort)].sort_values('state_name').reset_index(drop=True)
            labels=[]
            for i,row in sub.iterrows():
                label=row.state_name.title().replace('Nct Of Delhi','NCT of Delhi')
                geography = any(x in row.state_name.lower() for x in ['jammu','ladakh','dadra','daman'])
                if geography:label+=' †'
                labels.append(textwrap.fill(label,width=31))
                sparse=pd.notna(row['flags']) and ('sparse' in row['flags'] or 'effective_n' in row['flags'])
                if pd.isna(row.estimate):
                    ax.text(0,i,'Not estimable (private n=0)',fontsize=8,color='#6A6A6A',va='center')
                    continue
                if pd.notna(row.ci_lower):
                    ax.hlines(i,row.ci_lower,row.ci_upper,color=colors[wave],lw=1.25,alpha=.8)
                    ax.plot(row.estimate,i,'o',ms=5,mec=colors[wave],mfc='white' if sparse else colors[wave])
                else:
                    ax.plot(row.estimate,i,'x',ms=6,color=colors[wave])
                    ax.text(row.estimate+3,i,'CI withheld',fontsize=7,color='#666',va='center')
            ax.set_yticks(np.arange(len(sub)),labels,fontsize=9)
            ax.set_ylim(len(sub)-.35,-.85)
            ax.set_xlim(limits)
            ax.axvline(0,color='#555',lw=.8,linestyle='--')
            ax.set_xticks(np.arange(-40,101,20))
            ax.grid(axis='x',color='#E4E8EB',lw=.6)
            ax.set_axisbelow(True)
            ax.set_title(wave,fontweight='bold',fontsize=15,pad=14,color=colors[wave])
            ax.set_xlabel('Private minus public cesarean proportion (percentage points)',fontsize=10,labelpad=12)
            ax.tick_params(axis='y',length=0,pad=7)
            for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
        fig.suptitle('State/UT public–private cesarean delivery gaps',fontsize=20,fontweight='bold',y=.985)
        fig.text(.5,.957,heading+' | Survey-weighted descriptive estimates and approximate 95% intervals',ha='center',fontsize=11)
        handles=[Line2D([0],[0],marker='o',color='#555',linestyle='none',label='Estimate with interval'),
                 Line2D([0],[0],marker='o',mfc='white',mec='#555',color='#555',linestyle='none',label='Sparse count or weight-only effective n <30'),
                 Line2D([0],[0],marker='x',color='#555',linestyle='none',label='Point only; interval withheld')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.057),ncol=3,frameon=False,fontsize=9)
        fig.text(.06,.035,'† Wave-specific geography retained; these entries are not pooled or treated as equivalent across surveys.',fontsize=9)
        fig.text(.06,.018,'Facilities: m15=21 versus 31. Marginal Taylor t intervals; survey-design approximation applies. No covariate adjustment or change tests.',fontsize=9)
        fig.subplots_adjust(left=.19,right=.985,bottom=.12,top=.92,wspace=.70)
        stem=a.output/('statewise_gaps_'+cohort)
        fig.savefig(str(stem)+'.png',dpi=220,facecolor='white')
        fig.savefig(str(stem)+'.pdf',facecolor='white')
        plt.close(fig)


if __name__=='__main__':main()
