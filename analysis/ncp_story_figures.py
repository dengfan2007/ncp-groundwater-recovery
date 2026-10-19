"""Six-figure narrative: regional evidence, changing dates, and interpretation."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BLUE,RED,BLACK='#286090','#b33e36','#222222'


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    e=Path(a.evidence);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'outputs/review_revision/2026-10-15-r1'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':280})
    def save(fig,name):fig.savefig(out/name,bbox_inches='tight',facecolor='white');plt.close(fig)
    # Regional storage and a separate, explicitly common-window estimand.
    s=pd.read_csv(old/'deseasoned_series.csv',parse_dates=['date'])
    f=pd.read_csv(old/'fixed_window_spatial_contrasts.csv')
    f=f.loc[f.origin_dx.eq(0)&f.origin_dy.eq(0)&f.boundary.isna()]
    fig,axes=plt.subplots(2,1,figsize=(9,7),gridspec_kw={'height_ratios':[1.2,1]})
    for r,c in [('all',BLACK),('north',BLUE),('south',RED)]:
        g=s if r=='all' else s.loc[s.region_tag.eq(r)]
        m=g.groupby('date').deseasoned_gwa.mean();axes[0].plot(m.index,m,c=c,lw=1.15,label='All NCP' if r=='all' else r.capitalize())
    axes[0].axvspan(pd.Timestamp('2017-06-01'),pd.Timestamp('2018-05-01'),color='.9',zorder=-1)
    axes[0].set(ylabel='Deseasoned GWA (cm)',title='(a) Regional storage-proxy evolution');axes[0].legend(frameon=False,ncol=3,loc='lower left')
    for i,(start,method) in enumerate([('2019-01-01','OLS'),('2019-01-01','Theil-Sen'),('2022-01-01','OLS'),('2022-01-01','Theil-Sen')]):
        for size,c,dy in [(2.,BLUE,-.10),(3.,BLACK,.10)]:
            row=f.loc[f.start.eq(start)&f.method.eq(method)&f.block_degrees.eq(size)].iloc[0]
            axes[1].errorbar(row.estimate,i+dy,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt='o',color=c,ms=4,label=f'{int(size)}° blocks' if i==0 else None)
    axes[1].set(yticks=range(4),yticklabels=['2019–2024: OLS','2019–2024: Theil–Sen','2022–2024: OLS','2022–2024: Theil–Sen'],xlabel='South − North trend (cm/yr)',title='(b) Identical observation windows; all 27 cells')
    axes[1].invert_yaxis();axes[1].axvline(0,c='.6',ls=':');axes[1].legend(frameon=False,loc='lower right',fontsize=8)
    fig.tight_layout(h_pad=2);save(fig,'story_02_regional_evidence.png')
    # Endpoint counts and date ordering displayed on distinct axes.
    q=pd.read_csv(e/'actual_summary.csv');ends=['2019-12-01','2020-12-01','2021-12-01','2022-12-01','2023-12-01','2024-09-01']
    fig,axes=plt.subplots(1,2,figsize=(10,4.4));labels=['2019','2020','2021','2022','2023','2024*']
    for route,c,label in [('A',BLUE,'Deseasoned (A)'),('D',RED,'Reconstructed (D)')]:
        v=q.loc[q.route.eq(route)&q.rule.eq('highest_probability')&q.endpoint.isin(ends)].set_index('endpoint').reindex(ends)
        axes[0].plot(range(6),v.reportable,'o-',c=c,label=label)
        for rule,style in [('highest_probability','-'),('earliest','--')]:
            d=q.loc[q.route.eq(route)&q.rule.eq(rule)&q.endpoint.isin(ends)].set_index('endpoint').reindex(ends)
            axes[1].plot(range(6),d.timing_south_minus_north,marker='o',ms=4,ls=style,c=c,label=route+(': highest probability' if rule=='highest_probability' else ': earliest'))
    axes[0].set(ylabel='Cells meeting reporting rule',ylim=(-1,29),title='(a) Evidence changes as records extend');axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(ylabel='South − North date (months)',title='(b) Geographic ordering is conditional');axes[1].axhline(0,c='.6',ls=':');axes[1].legend(frameon=False,fontsize=8,loc='best')
    for ax in axes:ax.set(xticks=range(6),xticklabels=labels,xlabel='Record endpoint')
    fig.tight_layout(w_pad=2);save(fig,'story_03_endpoint_stability.png')
    # Representative selection was frozen before the comparison.
    s=pd.read_csv(e/'actual_series.csv',parse_dates=['date']);s=s.loc[s.endpoint.eq('2023-11-01')]
    cands=pd.read_csv(e/'actual_candidates.csv',parse_dates=['date']);cands=cands.loc[cands.endpoint.eq('2023-11-01')&cands.qualifies]
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.7),sharex=True)
    for row,(gid,label) in enumerate([('g_115.0_35.5','Southern cell (115.0, 35.5)'),('g_116.0_37.5','Northern cell (116.0, 37.5)')]):
        visible=s.loc[s.grid_id.eq(gid)&s.date.ge('2014-01-01')&s.route.isin(['A','D'])]
        limits=(visible[['observed','input_signal','beast_trend']].min().min()-2,visible[['observed','input_signal','beast_trend']].max().max()+2)
        for col,(route,color,name) in enumerate([('A',BLUE,'Deseasoned'),('D',RED,'Reconstructed')]):
            ax=axes[row,col];g=s.loc[s.grid_id.eq(gid)&s.route.eq(route)]
            ax.plot(g.date,g.observed,c='.82',lw=.7,label='Observed GWA')
            ax.plot(g.date,g.input_signal,c=color,lw=1,label=name+' input')
            ax.plot(g.date,g.beast_trend,c=BLACK,lw=1.1,label='BEAST mean trend')
            for d in cands.loc[cands.grid_id.eq(gid)&cands.route.eq(route),'date']:ax.axvline(d,c=color,ls=':',lw=.8)
            ax.set(ylabel='GWA (cm)',title=f'({chr(97+row*2+col)}) {label}\nRoute {route}: {name.lower()} signal',xlim=(pd.Timestamp('2014-01-01'),pd.Timestamp('2024-01-01')),ylim=limits)
            ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0,0].legend(frameon=False,fontsize=7,loc='upper right');axes[0,1].legend(frameon=False,fontsize=7,loc='upper right')
    for ax in axes[-1]:ax.set_xlabel('Year')
    fig.tight_layout();save(fig,'story_04_processing_comparison.png')
    # Simulation panels distinguish qualifying changes from sustained recovery.
    sm=pd.read_csv(e/'simulation_summary.csv');sel=pd.read_csv(e/'simulation_selections.csv');truth=pd.read_csv(e/'simulation_truth.csv');sel=sel.merge(truth,on='grid_id')
    scenarios=['no_recovery','temporary_gain','sustained_reversal','reversal_then_acceleration'];names=['No recovery','Temporary gain','Sustained reversal','Reversal + acceleration']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.7),gridspec_kw={'width_ratios':[1.1,1]})
    for j,(route,color) in enumerate([('A',BLUE),('D',RED)]):
        v=sm.loc[sm.route.eq(route)&sm.rule.eq('highest_probability')].set_index('scenario').reindex(scenarios)
        x=np.arange(4)+(j-.5)*.30
        axes[0].bar(x,v.qualification_rate,width=.28,color=color,label='Route '+route)
        axes[0].errorbar(x,v.qualification_rate,yerr=[v.qualification_rate-v.wilson_low,v.wilson_high-v.qualification_rate],fmt='none',ecolor=BLACK,lw=.8,capsize=2)
    axes[0].set(xticks=range(4),xticklabels=['No\nrecovery','Temporary\ngain','Sustained\nreversal','Reversal +\nacceleration'],ylabel='Fraction meeting reporting rule',ylim=(0,1.12),title='(a) Qualifying change is not proof of persistence')
    axes[0].legend(frameon=False,fontsize=8)
    positions=[];data=[];colors=[];labels2=[]
    for i,scenario in enumerate(['sustained_reversal','reversal_then_acceleration']):
        for j,(route,rule) in enumerate([('A','highest_probability'),('A','earliest'),('D','highest_probability'),('D','earliest')]):
            g=sel.loc[sel.scenario.eq(scenario)&sel.route.eq(route)&sel.rule.eq(rule)&sel.qualifies]
            dt=pd.to_datetime(g.selected_date);tr=pd.to_datetime(g.true_onset)
            errors=(dt.dt.year-tr.dt.year)*12+dt.dt.month-tr.dt.month
            data.append(errors);positions.append(i*5+j);colors.append(BLUE if route=='A' else RED);labels2.append(route+(' H' if rule=='highest_probability' else ' E'))
    bp=axes[1].boxplot(data,positions=positions,widths=.6,showfliers=False,patch_artist=True,medianprops={'color':BLACK})
    for box,c in zip(bp['boxes'],colors):box.set_facecolor(c);box.set_alpha(.65)
    axes[1].axhline(0,c=BLACK,ls=':',lw=1);axes[1].set(xticks=positions,xticklabels=labels2,ylabel='Selected date − true first reversal (months)',title='(b) Conditional timing error')
    axes[1].text(.18,-.19,'Sustained reversal',ha='center',transform=axes[1].transAxes,fontsize=9);axes[1].text(.76,-.19,'Reversal + acceleration',ha='center',transform=axes[1].transAxes,fontsize=9)
    axes[1].text(.5,-.28,'H: highest probability; E: earliest qualifying',ha='center',transform=axes[1].transAxes,fontsize=8)
    fig.tight_layout(w_pad=2);save(fig,'story_05_detection_scenarios.png')
    # Independent observation quantities, without treating head as storage.
    series=pd.read_csv(old/'aquifer_validation_series.csv',parse_dates=['date']);series=series.loc[series.support.eq('common')&series.date.between('2019-01-01','2024-08-01')]
    ci=pd.read_csv(e/'component_correlation_intervals.csv');ci=ci.loc[ci.aquifer.eq('unconfined')&ci.treatment.eq('difference')&ci.metric.isin(['TWSA','SMSA','GWA'])]
    fig,axes=plt.subplots(3,1,figsize=(9,8),gridspec_kw={'height_ratios':[1,1,.9]})
    for ax,r,letter in zip(axes[:2],['north','south'],'ab'):
        for aq,col,c,label in [('unconfined','gwa_cm',BLACK,'GWA'),('unconfined','well_anomaly',BLUE,'Unconfined wells'),('confined','well_anomaly',RED,'Confined wells')]:
            g=series.loc[series.region_tag.eq(r)&series.aquifer.eq(aq)];z=(g[col]-g[col].mean())/g[col].std();ax.plot(g.date,z,c=c,lw=1,label=label)
        ax.set(ylabel='Standardized anomaly',title=f'({letter}) {r.capitalize()}: same cells, fixed well cohorts')
    axes[0].legend(frameon=False,ncol=3,fontsize=8,loc='lower right')
    for j,(r,c) in enumerate([('north',BLUE),('south',RED)]):
        v=ci.loc[ci.region.eq(r)].set_index('metric').reindex(['TWSA','SMSA','GWA']);x=np.arange(3)+(j-.5)*.12
        axes[2].errorbar(x,v.estimate,yerr=[v.estimate-v.low,v.high-v.estimate],fmt='o',c=c,ms=5,capsize=3,label=r.capitalize())
    axes[2].set(xticks=range(3),xticklabels=['TWSA','SMSA (subtracted)','GWA'],ylabel='Monthly-change correlation',ylim=(-1,1),title='(c) Component subtraction changes shallow-well covariation');axes[2].axhline(0,c='.6',ls=':');axes[2].legend(frameon=False,ncol=2,fontsize=8)
    fig.tight_layout(h_pad=1.6);save(fig,'story_06_observation_support.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    texts=[('REGIONAL EVIDENCE','Storage-proxy rise in the early 2020s\nHigher southern trend estimates\nPublic well levels also rise\nStorage and head remain distinct'),('TIMING INTERPRETATION','Dates change with record endpoints\nReconstruction changes selected features\nHighest probability need not mean onset\nShort-lived gains can meet the rule'),('MONITORING IMPLICATION','Report trend and date stability separately\nPreserve observation support\nTest for temporary gains\nTreat precise onset sequences as conditional')]
    for ax,(title,body) in zip(axes,texts):ax.axis('off');ax.text(.02,.85,title,c=BLUE,fontweight='bold',fontsize=13);ax.text(.02,.67,body,va='top',fontsize=10.2,linespacing=2.)
    fig.tight_layout();save(fig,'graphical_abstract_1328x531.png')


if __name__=='__main__':main()
