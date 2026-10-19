"""Render the review revision from saved numerical evidence."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BLUE,RED,BLACK='#286090','#b33e36','#222222'


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    e=Path(a.evidence);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':280})
    def save(fig,name):
        fig.savefig(out/name,bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,4.3));ax.axis('off')
    boxes=[(.01,.61,'Monthly observations\n27 cells; 2003–2024\nGWA residual storage proxy\nPotential surface-water contribution'),(.355,.61,'Recovery-related candidates\nCalendar gaps retained\nBEAST and one-break models\nHighest probability vs earliest'),(.70,.61,'Retrospective detection\nAnnual record endpoints\nSeasonality refitted at each cutoff\nFirst report vs selected date'),(.01,.08,'Spatial contrasts\nCommon windows; all 27 cells\nBlock scales, origins and boundaries\nOLS and Theil–Sen trends'),(.355,.08,'Observation comparisons\nFixed unconfined / confined cohorts\n18 cells shared by both cohorts\nStorage-component diagnostics'),(.70,.08,'Trajectory interpretation\nSlope changes vs level shifts\nPaired gap simulations\nClimate and surface-water context')]
    for x,y,txt in boxes:
        ax.add_patch(Rectangle((x,y),.29,.31,fill=False,edgecolor=BLUE,lw=1.2));ax.text(x+.145,y+.155,txt,ha='center',va='center',linespacing=1.6,fontsize=9.5)
    for x in [.315,.66]:ax.annotate('',xy=(x+.035,.76),xytext=(x,.76),arrowprops=dict(arrowstyle='->',color=BLACK))
    for x in [.155,.5,.845]:ax.annotate('',xy=(x,.41),xytext=(x,.58),arrowprops=dict(arrowstyle='->',color=BLACK))
    save(fig,'figure_02_workflow.png')
    s=pd.read_csv(e/'deseasoned_series.csv',parse_dates=['date'])
    fig,axes=plt.subplots(2,1,figsize=(8.7,6),sharex=True)
    for r,color in [('all',BLACK),('north',BLUE),('south',RED)]:
        g=s if r=='all' else s.loc[s.region_tag.eq(r)]
        means=g.groupby('date')[['gwa_cm','deseasoned_gwa']].mean()
        for ax,col in zip(axes,['gwa_cm','deseasoned_gwa']):ax.plot(means.index,means[col],c=color,lw=1,label='All NCP' if r=='all' else r.capitalize())
    for ax,title in zip(axes,['(a) Observed monthly storage proxy','(b) Seasonal component removed; gaps retained']):
        ax.set(ylabel='GWA (cm)',title=title);ax.axvspan(pd.Timestamp('2017-06-01'),pd.Timestamp('2018-05-01'),color='.9',zorder=-2);ax.axvline(pd.Timestamp('2014-12-01'),ls=':',c='.5')
    axes[0].legend(ncol=3,frameon=False,loc='lower left');axes[-1].set_xlabel('Year');fig.tight_layout();save(fig,'figure_03_regional_gwa_timeseries.png')
    f=pd.read_csv(e/'fixed_window_spatial_contrasts.csv');f=f.loc[f.origin_dx.eq(0)&f.origin_dy.eq(0)&f.boundary.isna()]
    b=pd.read_csv(ROOT/'outputs/calendar_revision/2026-10-12-r4/breakpoints_calendar_stl.csv',parse_dates=['onset_date']).dropna(subset=['onset_date'])
    fig,axes=plt.subplots(1,2,figsize=(9,4.5),gridspec_kw={'width_ratios':[1,1.25]})
    rng=np.random.default_rng(42)
    for i,r in enumerate(['north','south']):
        v=b[b.region_tag.eq(r)];axes[0].scatter(v.onset_date,i+rng.uniform(-.10,.10,len(v)),s=25,c=[BLUE,RED][i])
    axes[0].set(yticks=[0,1],yticklabels=['North (8)','South (16)'],ylim=(-.5,1.5),title='(a) Selected dates',xlabel='Year')
    axes[0].xaxis.set_major_locator(mdates.YearLocator(2));axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    labels=[]
    for i,(start,method) in enumerate([('2019-01-01','OLS'),('2019-01-01','Theil-Sen'),('2022-01-01','OLS'),('2022-01-01','Theil-Sen')]):
        labels.append(f'{start[:4]}–2024\n{method}')
        for size,color,dy in [(2.,BLUE,-.10),(3.,BLACK,.10)]:
            row=f.loc[f.start.eq(start)&f.method.eq(method)&f.block_degrees.eq(size)].iloc[0]
            axes[1].errorbar(row.estimate,i+dy,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt='o',color=color,ms=4,lw=1.3,label=f'{int(size)}° blocks' if i==0 else None)
    axes[1].set(yticks=range(4),yticklabels=labels,xlabel='South − North trend (cm/yr)',title='(b) Common windows, all 27 cells');axes[1].invert_yaxis();axes[1].axvline(0,c='.6',ls=':');axes[1].legend(frameon=False,fontsize=8,loc='lower right');fig.tight_layout(w_pad=2);save(fig,'figure_04_recovery_metrics.png')
    r=pd.read_csv(e/'rolling_breakpoints.csv',parse_dates=['onset_date']);ends=r.endpoint.unique();ids=b.sort_values(['region_tag','onset_date']).grid_id.tolist()
    # Include cells unassigned in the final record, rather than hiding reversals.
    ids+=sorted(set(r.grid_id)-set(ids))
    matrix=np.full((len(ids),len(ends)),np.nan)
    for i,gid in enumerate(ids):
        for j,end in enumerate(ends):
            row=r.loc[r.grid_id.eq(gid)&r.endpoint.eq(end)].iloc[0]
            if pd.notna(row.onset_date):matrix[i,j]=row.onset_date.year+(row.onset_date.month-1)/12
    summary=pd.read_csv(e/'rolling_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,7),gridspec_kw={'width_ratios':[1,1.8]})
    axes[0].plot(range(len(ends)),summary.n_reportable,'o-',c=BLACK);axes[0].set(xticks=range(len(ends)),xticklabels=[x[:4] for x in ends],ylabel='Cells meeting reporting rule',xlabel='Record endpoint',ylim=(0,28),title='(a) Number detected')
    for j,n in enumerate(summary.n_reportable):axes[0].annotate(str(n),(j,n),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('.90')
    im=axes[1].imshow(matrix,aspect='auto',cmap=cmap,vmin=2015,vmax=2022.5)
    axes[1].set(yticks=range(len(ids)),yticklabels=[x.replace('g_','').replace('_',', ') for x in ids],xticks=range(len(ends)),xticklabels=[x[:4] for x in ends],xlabel='Record endpoint',title='(b) Selected date changes with the record')
    axes[1].tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=axes[1],fraction=.045,pad=.025,label='Selected year')
    axes[1].text(0,-.12,'Gray: no qualifying date. Rows: cell lower-left coordinates.',transform=axes[1].transAxes,fontsize=8)
    fig.tight_layout(w_pad=2);save(fig,'figure_07_detection_stability.png')
    shapes=pd.read_csv(e/'shape_fitted_series.csv',parse_dates=['date']);cand=pd.read_csv(e/'beast_candidates.csv',parse_dates=['date'])
    fig,axes=plt.subplots(2,1,figsize=(8.7,6.2),sharex=True)
    for ax,gid,title in zip(axes,['g_115.0_35.5','g_116.0_37.5'],['(a) Southern cell: later rise and an earlier candidate','(b) Northern cell: decline followed by improvement']):
        sub=shapes.loc[shapes.grid_id.eq(gid)]
        raw=sub.loc[sub.model.eq('linear')];ax.plot(raw.date,raw.observed,c='.65',lw=.8,label='Observed')
        for model,label,color,style in [('hinge','Continuous slope change',BLUE,'-'),('step_hinge','Level + slope change',RED,'--')]:
            v=sub[sub.model.eq(model)];ax.plot(v.date,v.fitted,c=color,ls=style,lw=1.3,label=label)
        v=cand.loc[cand.grid_id.eq(gid)&cand.qualifies]
        for d in v.date:ax.axvline(d,c=BLACK,ls=':',lw=.8)
        ax.set(ylabel='GWA (cm)',title=title+'\n'+gid.replace('g_','Cell ').replace('_',', '))
    axes[0].legend(frameon=False,ncol=3,fontsize=8,loc='upper left');axes[-1].set_xlabel('Year');fig.tight_layout();save(fig,'figure_08_change_shapes.png')
    s=pd.read_csv(e/'aquifer_validation_series.csv',parse_dates=['date']);s=s.loc[s.support.eq('common')]
    m=pd.read_csv(e/'aquifer_validation_metrics.csv');m=m.loc[m.support.eq('common')&m.period.eq('2019+')]
    fig,axes=plt.subplots(2,1,figsize=(8.7,6),sharex=True)
    for ax,region,letter in zip(axes,['north','south'],'ab'):
        for aquifer,column,label,color in [('unconfined','gwa_cm','GWA',BLACK),('unconfined','well_anomaly','Unconfined wells',BLUE),('confined','well_anomaly','Confined wells',RED)]:
            g=s.loc[s.region_tag.eq(region)&s.aquifer.eq(aquifer)&s.date.ge('2019-01-01')]
            z=(g[column]-g[column].mean())/g[column].std();ax.plot(g.date,z,color=color,lw=1,label=label)
        v=m.loc[m.region.eq(region)];ru=v.loc[v.aquifer.eq('unconfined'),'pearson'].iloc[0];rc=v.loc[v.aquifer.eq('confined'),'pearson'].iloc[0]
        ax.set(ylabel='Standardized anomaly',title=f'({letter}) {region.capitalize()} | raw r: unconfined {ru:.2f}, confined {rc:.2f}')
    axes[0].legend(ncol=3,frameon=False,fontsize=9,loc='lower right');axes[-1].set_xlabel('Year');fig.tight_layout();save(fig,'figure_09_validation.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    texts=[('OBSERVATIONS','27 cells; monthly storage proxy\nCalendar gaps retained\nUnconfined and confined wells\nCommon spatial and temporal support'),('TIMING','24 full-record selected breaks\nDates migrate as records extend\nCandidate selection changes ordering\nRetrospective transitions, not onset clocks'),('REGIONAL CONTRASTS','Higher southern trend estimates\nIntervals depend on scale and window\nWeak northern agreement in both aquifers\nLand-water subtraction affects covariation')]
    for ax,(title,body) in zip(axes,texts):ax.axis('off');ax.text(.02,.85,title,c=BLUE,fontweight='bold',fontsize=14);ax.text(.02,.67,body,va='top',fontsize=10.5,linespacing=2.)
    fig.tight_layout();save(fig,'graphical_abstract_1328x531.png')


if __name__=='__main__':main()
