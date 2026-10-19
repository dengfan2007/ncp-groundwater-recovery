"""Render-only figures for declared diagnostics and completed simulations."""
from __future__ import annotations
import argparse
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import ncp_figure_upgrade as old
from ncp_monitoring_core import ROOT,STORY,REVIEW,VALIDATION,EVIDENCE,RUNTIME,BOUNDARY,seed,sha,json_write
from ncp_monitoring_diagnostics import spatial_weights,quantiles

BLACK,GRAY,BLUE,RED=old.BLACK,old.GRAY,old.BLUE,old.RED
LOWLAND_CONTEXT=ROOT/'manuscripts/assets/jhrs-v30/figure_01_lowland_mask_200m.geojson'
LOWLAND_OUTLINE='#c4cbd0'

class Figures(old.Renderer):
    def __init__(self,out):
        super().__init__(SimpleNamespace(evidence=str(VALIDATION),story=str(STORY),review=str(REVIEW),output_dir=str(out),baseline_dir=None))
        self.new=EVIDENCE;self.supplement=[];self._lowland_context=None
    def nr(self,name,**kwargs):return self.read(self.new/name,**kwargs)
    def save(self,fig,name):
        # Matplotlib rasterizes long colorbar meshes by default. Keep all
        # scientific shapes and their scale legends vector in exported PDFs.
        for ax in fig.axes:
            for artist in ax.collections:
                artist.set_rasterized(False)
                if artist.__class__.__name__=='QuadMesh' and np.size(artist.get_array())>=128:
                    artist.set_edgecolor('face')
        super().save(fig,name)
    def supplement_save(self,fig,name,caption):
        self.save(fig,name);self.supplement.append(dict(name=name,caption=caption))
    def lowland_context(self):
        """Return the frozen, display-only <=200 m mask used in Figure 1.

        The grid definition and scientific analysis remain unchanged.  This
        geometry supplies geographic orientation only, so it is intentionally
        drawn beneath the aggregation cells and well markers.
        """
        if self._lowland_context is None:
            if not LOWLAND_CONTEXT.is_file():
                raise FileNotFoundError(f'Missing Figure 1 lowland context: {LOWLAND_CONTEXT}')
            self.inputs[LOWLAND_CONTEXT.relative_to(ROOT).as_posix()]=sha(LOWLAND_CONTEXT)
            context=json.loads(LOWLAND_CONTEXT.read_text(encoding='utf-8'))
            if context.get('type')!='FeatureCollection':
                raise ValueError('Figure 1 lowland context must be a GeoJSON FeatureCollection.')
            self._lowland_context=[f['geometry'] for f in context.get('features',[]) if f.get('geometry')]
        return self._lowland_context
    @staticmethod
    def _context_polygons(geometry):
        if geometry['type']=='Polygon':
            return [geometry['coordinates']]
        if geometry['type']=='MultiPolygon':
            return geometry['coordinates']
        raise ValueError(f"Unsupported Figure 1 context geometry: {geometry['type']}")
    def draw_lowland_context(self,ax):
        for geometry in self.lowland_context():
            for polygon in self._context_polygons(geometry):
                if not polygon or len(polygon[0])<4:
                    continue
                exterior=np.asarray(polygon[0],dtype=float)
                # Display only the generalized exterior.  Fine mask holes are
                # analytically retained in cell coverage but would become
                # visual noise at the manuscript's printed figure size.
                ax.plot(exterior[:,0],exterior[:,1],color=LOWLAND_OUTLINE,lw=.48,zorder=-1)
    def grid(self,ax,coverage=None,region_fill=True,transparent_cells=False):
        super().grid(ax,coverage,region_fill)
        if transparent_cells:
            # In panel (b), white cell fills would mask the contextual
            # lowland outline. Keep only cell edges and well symbols there.
            for patch in ax.patches:
                if patch.get_fill():
                    patch.set_facecolor('none')
                    patch.set_alpha(0)
    def wells(self):
        self.grids=self.read(ROOT/'outputs/figure_tables_en/fig01_grid_map.csv').drop_duplicates('grid_id');self.meta=self.grids.set_index('grid_id')
        m=self.read(REVIEW/'well_cohort_membership.csv');raw=self.read(ROOT/'data/external/ncp_long2025_well_2005_2024.csv')
        coords=raw.loc[raw.grid_id.isin(self.grids.grid_id)].drop_duplicates('well_id')[['well_id','lon','lat','aquifer_group','grid_id']]
        self.common=set(m.loc[m.aquifer_group.eq('unconfined'),'grid_id'])&set(m.loc[m.aquifer_group.eq('confined'),'grid_id'])
        self.fixed=coords.loc[coords.well_id.isin(m.well_id)&coords.grid_id.isin(self.common)];self.coords=coords
        self.table('fig01_fixed_wells',self.fixed);self.table('fig01_analysis_cells',self.grids)
        fig,axs=self.subplots(1,2,112)
        for ax in axs:
            self.draw_lowland_context(ax)
            self.grid(ax,self.common if ax==axs[1] else None,ax==axs[0],transparent_cells=ax==axs[1])
        axs[0].set_title('(a) Domain and regional convention',loc='left');axs[1].set_title('(b) Matched fixed well cohorts',loc='left')
        for gid in BOUNDARY:
            _,x,y=gid.split('_');axs[0].add_patch(Rectangle((float(x),float(y)),1,1,fill=False,hatch='///',edgecolor='.55',lw=.4))
        axs[0].text(.03,.98,'North: 11 cells\nSouth: 16 cells',transform=axs[0].transAxes,va='top',fontsize=8.5)
        axs[0].scatter([116.4,117.2],[39.9,39.1],c=BLACK,s=8)
        axs[0].annotate('Beijing',(116.4,39.9),xytext=(7,3),textcoords='offset points',fontsize=8.5);axs[0].annotate('Tianjin',(117.2,39.1),xytext=(7,-8),textcoords='offset points',fontsize=8.5)
        well_styles={
            'unconfined':dict(s=4.8,marker='o',facecolors=BLACK,edgecolors=BLACK,linewidths=.22,alpha=.40),
            'confined':dict(s=9,marker='^',facecolors='none',edgecolors=BLACK,linewidths=.48,alpha=.76),
        }
        # Stable mixed drawing order avoids giving either aquifer a fixed
        # foreground layer. Keep original coordinates, including co-locations.
        ordered=self.fixed.sort_values('well_id').sample(frac=1,random_state=20261019)
        for well in ordered.itertuples():
            axs[1].scatter(well.lon,well.lat,zorder=2,**well_styles[well.aquifer_group])
        handles=[];labels=[]
        for aq,label in [('unconfined','Unconfined'),('confined','Confined')]:
            handles.append(axs[1].scatter([],[],**well_styles[aq]))
            labels.append(f'{label}: {self.fixed.aquifer_group.eq(aq).sum()}')
        fig.legend(handles,labels,loc='outside lower center',frameon=False,fontsize=8.5,markerscale=1.3,ncol=2)
        axs[1].text(.03,.98,'18 common cells\n10 north\n8 south',transform=axs[1].transAxes,va='top',fontsize=8.5)
        self.save(fig,'figure_01_study_area')
    def trends(self):
        source=self.read(REVIEW/'deseasoned_series.csv',parse_dates=['date']);series=[]
        fig,axs=self.subplots(2,1,180,gridspec_kw={'height_ratios':[1.1,1]})
        for region,col,label in [('all',BLACK,'All cells'),('north',BLUE,'North'),('south',RED,'South')]:
            g=source if region=='all' else source.loc[source.region_tag.eq(region)];v=g.groupby('date').deseasoned_gwa.mean().loc[:'2024-08-01'];axs[0].plot(v.index,v,c=col,label=label);series.extend(pd.DataFrame(dict(date=v.index,region=region,value=v.values)).to_dict('records'))
        gap_style=dict(facecolor='.96',edgecolor='.68',hatch='////',linewidth=0)
        wet_style=dict(facecolor='.88',edgecolor='none')
        axs[0].axvspan(pd.Timestamp('2017-07-01'),pd.Timestamp('2018-05-01'),zorder=0,**gap_style)
        for start,end in [('2021-07-01','2022-06-01'),('2023-07-01','2023-12-01')]:axs[0].axvspan(pd.Timestamp(start),pd.Timestamp(end)+pd.DateOffset(months=1),zorder=0,**wet_style)
        events=[('2014-06-01','Hebei pilot\nJun 2014',.97,'right'),
                ('2014-12-01','Water diversion\nDec 2014',.82,'right'),
                ('2018-09-01','River replenishment\nSep 2018',.97,'left'),
                ('2019-01-01','Regional action plan\nJan 2019',.82,'left')]
        for date,label,level,align in events:
            d=pd.Timestamp(date);axs[0].axvline(d,c='.6',lw=.6,ls=':');axs[0].annotate(label,xy=(d,level),xycoords=('data','axes fraction'),xytext=(-3 if align=='right' else 3,0),textcoords='offset points',va='top',ha=align,fontsize=8.5)
        axs[0].set(ylabel='Deseasoned GWA (cm)',title='(a) Regional trajectories and contextual events')
        handles,labels=axs[0].get_legend_handles_labels()
        handles.extend([Patch(**gap_style),Patch(**wet_style)])
        labels.extend(['GRACE/GRACE-FO gap','Wet-period windows'])
        axs[0].legend(handles,labels,frameon=False,loc='lower left',ncol=2,fontsize=8.5,columnspacing=1,handlelength=2)
        choices=[('archived','original','complete','Original grouping'),('archived','flip','complete','Boundary row north'),('archived','exclude','complete','Boundary row excluded'),('archived','original','end_2023_06','End June 2023'),('archived','original','exclude_both','Both event windows\nexcluded'),('shared_noah','original','complete','Noah: common support'),('era2','original','complete','ERA5-Land: 0–2 m'),('era289','original','complete','ERA5-Land: 0–2.89 m')]
        plotted=[]
        for y,(model,boundary,version,label) in enumerate(choices):
            d=self.nr(f'diagnostics/{model}/trend_contrasts.csv');r=d.loc[d.start.eq('2019-01-01')&d.boundary.eq(boundary)&d.version.eq(version)&d.method.eq('OLS')&d.block_degrees.eq(3)].iloc[0]
            axs[1].errorbar(r.estimate,y,xerr=[[r.estimate-r.low],[r.high-r.estimate]],fmt='o',c=BLACK if y<5 else GRAY,capsize=2,ms=4);plotted.append({**r.to_dict(),'label':label})
            if y==0:
                value=f'{r.estimate:.2f} [{r.low:.2f}, {r.high:.2f}]'.replace('-', '\N{MINUS SIGN}')
                axs[1].annotate(value,xy=(r.estimate,y),xytext=(0,7),textcoords='offset points',ha='center',va='bottom',fontsize=8.5,fontweight='bold')
        axs[1].set_yticks(range(len(choices)),[c[3] for c in choices]);axs[1].invert_yaxis();axs[1].axvline(0,c=BLACK,ls='--',lw=1.05,zorder=0)
        axs[1].set_ylim(len(choices)-.6,-.9)
        # Rows 0-4 use the archived domain; rows 5-7 share one land-model support.
        axs[1].axhline(4.5,c='.75',lw=.6)
        for y,label in [(-.5,'Archived GWA'),(4.55,'Common land-model support')]:axs[1].text(.01,y,label,transform=axs[1].get_yaxis_transform(),ha='left',va='top',fontsize=8.5,color='.4')
        axs[1].set(xlabel='South − north OLS trend (cm/yr)',title='(b) 2019+ estimates: 3° joint 95% intervals')
        self.table('fig02_regional_series',pd.DataFrame(series));self.table('fig02_sensitivities',pd.DataFrame(plotted));self.save(fig,'story_02_regional_evidence')
    def endpoints(self):
        d=self.nr('inference/real_selections.csv',parse_dates=['selected_date','endpoint']);meta=self.meta
        contrasts=self.nr('inference/endpoint_date_contrasts.csv',parse_dates=['endpoint'])
        fig=plt.figure(figsize=(160*old.MM,200*old.MM),layout='constrained');gs=fig.add_gridspec(3,2,height_ratios=[.8,1,1.25])
        axs=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:])];paired=[fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])]
        for route,col,marker in [('A',BLACK,'o'),('D',GRAY,'^')]:
            # No calibrated rule passed validation, so only original-rule counts exist.
            q=d.loc[d.route.eq(route)&d.rule.eq('highest_probability')&d.setting.eq('original')]
            counts=q.groupby('endpoint').qualifies.sum();axs[0].plot(counts.index,counts.values,c=col,marker=marker,label=f'Route {route}')
            for r in contrasts.loc[contrasts.route.eq(route)].itertuples():
                offset=pd.DateOffset(days=-15 if route=='A' else 15);axs[1].errorbar(r.endpoint+offset,r.estimate,yerr=[[r.estimate-r.low],[r.high-r.estimate]],fmt=marker,c=col,capsize=2,ms=4)
        axs[0].axhline(27,c='.6',ls=':',lw=.8);axs[0].set(ylabel='Cells with a date',ylim=(-1,29),title='(a) Cells with a qualifying date');axs[0].legend(frameon=False,ncol=2,fontsize=8.5,loc='lower right')
        axs[1].axhline(0,c='.6',ls=':');axs[1].set(ylabel='South − north (months)',title='(b) Regional date contrast with 3° spatial-block intervals')
        for ax in axs:
            ends=sorted(d.endpoint.unique());ax.set_xticks(ends,[pd.Timestamp(x).strftime('%b\n%Y') for x in ends]);ax.set_xlabel('Observed record endpoint')
        original=d.loc[d.setting.eq('original')&d.rule.eq('highest_probability')];xx=np.array([2014.5,2023.5])
        for ax,(route,marker,panel) in zip(paired,[('A','o','c'),('D','^','d')]):
            a=original.loc[original.route.eq(route)&original.endpoint.eq(pd.Timestamp('2023-12-01'))&original.qualifies];b=original.loc[original.route.eq(route)&original.endpoint.eq(pd.Timestamp('2024-08-01'))&original.qualifies]
            m=a.merge(b,on='grid_id');x=m.selected_date_x.dt.year+(m.selected_date_x.dt.month-1)/12;y=m.selected_date_y.dt.year+(m.selected_date_y.dt.month-1)/12
            stable=int(((m.selected_date_y.dt.year-m.selected_date_x.dt.year)*12+m.selected_date_y.dt.month-m.selected_date_x.dt.month).abs().le(6).sum())
            ax.plot(xx,xx,c='.4',lw=.8);ax.fill_between(xx,xx-.5,xx+.5,color='.9',zorder=0)
            for region,col in [('north',BLUE),('south',RED)]:
                ok=m.region_tag_x.eq(region);ax.scatter(x[ok],y[ok],c=col,marker=marker,s=24,alpha=.8,edgecolors='white',linewidths=.4,label=region.capitalize())
            ax.text(.97,.04,f'{stable}/{len(m)} within ±6 months',transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5)
            ax.set(xlim=xx,ylim=xx,xlabel='Date at Dec 2023 endpoint',title=f'({panel}) Route {route}: paired dates');ax.set_aspect('equal')
        paired[0].set_ylabel('Date at Aug 2024 endpoint');paired[0].legend(frameon=False,fontsize=8.5,loc='lower left',bbox_to_anchor=(0,.12))
        self.table('fig03_endpoint_contrasts',contrasts);self.table('fig03_dates',d);self.save(fig,'validation_03_endpoint_stability')
    def processing(self):
        s=self.nr('processing/series.csv',parse_dates=['date']);c=self.nr('processing/candidates.csv');picks=self.nr('processing/selections.csv')
        self.table('fig04_signals',s);self.table('fig04_candidates',c);self.table('fig04_selections',picks)
        fig=plt.figure(figsize=(160*old.MM,188*old.MM),layout='constrained');outer=fig.add_gridspec(2,2)
        for i,gid in enumerate(['g_115.0_35.5','g_116.0_37.5']):
            visible=s.loc[s.grid_id.eq(gid)&s.date.ge('2014-01-01')];ylow=visible[['observed','input_signal','fitted_trend']].min().min()-2;yhigh=visible[['observed','input_signal','fitted_trend']].max().max()+2
            for j,(route,col) in enumerate([('A',BLACK),('D',GRAY)]):
                inner=outer[i,j].subgridspec(2,1,height_ratios=[4,1],hspace=.02);ax=fig.add_subplot(inner[0]);prob=fig.add_subplot(inner[1],sharex=ax)
                g=s.loc[s.grid_id.eq(gid)&s.route.eq(route)&s.date.ge('2014-01-01')]
                ax.scatter(g.date,g.observed,s=4,c='.7',linewidths=0);ax.plot(g.date,g.input_signal,c=col,lw=1.2);ax.plot(g.date,g.fitted_trend,c=BLACK,ls='--',lw=.8)
                for a in [ax,prob]:a.axvspan(pd.Timestamp('2017-07-01'),pd.Timestamp('2018-05-01'),color='.9',zorder=0)
                pp=picks.loc[picks.grid_id.eq(gid)&picks.route.eq(route)&picks.qualifies]
                for date in pp.selected_date.dropna().unique():
                    rules=pp.loc[pp.selected_date.eq(date),'rule'].tolist();label='H=E' if len(rules)>1 else ('H' if rules[0]=='highest_probability' else 'E')
                    ax.axvline(pd.Timestamp(date),c=BLACK,ls='--' if 'highest_probability' in rules else '-.',lw=.8);ax.text(pd.Timestamp(date),.97,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8.5)
                prob.plot(g.date,g.cp_probability,c=col,lw=.8);prob.set(ylim=(0,1.05),ylabel='Pr(CP)');prob.set_yticks([0,1]);prob.tick_params(labelsize=8.5)
                ax.set(title=f'({chr(97+i*2+j)}) '+('Southern' if i==0 else 'Northern')+f' cell / {route}',ylabel='GWA (cm)',ylim=(ylow,yhigh));ax.tick_params(labelbottom=False)
                prob.xaxis.set_major_locator(mdates.YearLocator(4));prob.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        self.save(fig,'story_04_processing_comparison')
        for page,ids in enumerate([self.grids.grid_id.tolist()[:14],self.grids.grid_id.tolist()[14:]],1):
            fig,axs=plt.subplots(7,2,figsize=(160*old.MM,215*old.MM),layout='constrained',sharex=True)
            for ax,gid in zip(axs.flat,ids):
                for route,col,ls in [('A',BLACK,'-'),('D',GRAY,'--')]:
                    g=s.loc[s.grid_id.eq(gid)&s.route.eq(route)];ax.plot(g.date,g.input_signal,c=col,ls=ls,lw=.6)
                ax.axvspan(pd.Timestamp('2017-07-01'),pd.Timestamp('2018-05-01'),color='.9');ax.set_title(gid,fontsize=8.5,loc='left');ax.tick_params(labelsize=8.5)
                ax.xaxis.set_major_locator(mdates.YearLocator(10));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            for ax in list(axs.flat)[len(ids):]:ax.set_visible(False)
            self.supplement_save(fig,f'monitoring_all_cells_{page}',f'All retained cells, page {page}. Black solid: A; gray dashed: D. Inputs fitted through November 2023; shading marks the mission gap. These are aggregation units, not independent satellite footprints.')
    def simulation(self):
        summary=self.nr('inference/two_stage_summary.csv');baseline=self.nr('inference/original_rule_validation.csv');trials=self.nr('inference/two_stage_trials.csv')
        self.table('fig05_summary',summary);self.table('fig05_false_reports',baseline)
        fig,axs=self.subplots(2,2,172)
        dates=pd.date_range('2014-01-01','2024-08-01',freq='MS');t=(dates.year-2003)+(dates.month-1)/12
        for weak,ls in [(.3,':'),(.6,'-'),(1.2,'--')]:
            curve=-1.5*t+(1.5+weak)*np.maximum(t-15,0)+(4-weak)*np.maximum(t-18,0);axs[0,0].plot(dates,curve,c=BLACK,ls=ls,label=f'Weak slope +{weak:g}')
        axs[0,0].axvspan(pd.Timestamp('2017-07-01'),pd.Timestamp('2018-05-01'),color='.9');axs[0,0].axvline(pd.Timestamp('2021-01-01'),c='.5',ls=':');axs[0,0].set(title='(a) Two-stage truths: 36-month separation',ylabel='True storage trend (cm)');axs[0,0].legend(frameon=False,fontsize=8.5);axs[0,0].xaxis.set_major_locator(mdates.YearLocator(4));axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        search=self.nr('inference/calibration_search.csv');strict=search.loc[search.probability_threshold.eq(.99)&search.gain_threshold.eq(2.0)]
        nulls=[(s,r) for s in ['no_recovery','temporary_gain'] for r in ['low','median','high']]
        for j,(route,col,marker) in enumerate([('A',BLACK,'o'),('D',GRAY,'^')]):
            for k,(source,filled) in enumerate([(baseline,True),(strict,False)]):
                g=source.loc[source.route.eq(route)].set_index(['scenario','regime']).reindex(nulls);x=np.arange(6)+(j*2+k-1.5)*.13
                axs[0,1].errorbar(x,g.fraction,yerr=[g.fraction-g.low,g.high-g.fraction],fmt=marker,c=col,mfc=col if filled else 'white',capsize=1.5,ms=3.5,lw=.8,
                                  label=f'{route}: '+('original rule' if filled else 'strictest grid'))
        axs[0,1].axhline(.05,c='.5',ls=':');axs[0,1].text(5.6,.055,'5%',fontsize=8.5,color='.4',va='bottom',ha='right');axs[0,1].axvline(2.5,c='.85',lw=.6)
        axs[0,1].set_xticks(range(6),['L','M\nDecline','H','L','M\nTemp. gain','H']);axs[0,1].set_xlabel('Null trajectory and residual level')
        axs[0,1].set(title='(b) False reports under null trajectories',ylabel='Fraction of 500 realizations',ylim=(0,.62));axs[0,1].legend(frameon=False,fontsize=8.5,ncol=2,loc='upper left',handlelength=1,columnspacing=.6,handletextpad=.3)
        labels=[];boxes=[];colors=[];box_rows=[]
        for route,col in [('A',BLACK),('D',GRAY)]:
            for mask in ['complete','observed']:
                q=trials.loc[trials.route.eq(route)&trials.setting.eq('original')&trials.rule.eq('highest_probability')&trials.regime.eq('median')&trials.onset.eq('2018-01-01')&trials.weak_slope.eq(.6)&trials['mask'].eq(mask)].error_months.dropna().to_numpy();boxes.append(q);labels.append(route+'\n'+('Full' if mask=='complete' else 'Gaps'));colors.append(col)
                box_rows.extend(dict(route=route,mask=mask,error_months=float(value)) for value in q)
        self.table('fig05_conditional_errors',pd.DataFrame(box_rows))
        bp=axs[1,0].boxplot(boxes,patch_artist=True,showfliers=True,widths=.55,medianprops=dict(color=BLACK,lw=1),flierprops=dict(marker='.',markersize=2,alpha=.5))
        for box,col in zip(bp['boxes'],colors):box.set(facecolor=col,alpha=.3)
        axs[1,0].axhline(0,c='.5',ls=':');axs[1,0].axhline(36,c='.5',ls='--');axs[1,0].set_xticks(range(1,5),labels)
        for level,label in [(0,'first\nreversal'),(36,'accel-\neration')]:axs[1,0].text(1.01,level,label,transform=axs[1,0].get_yaxis_transform(),ha='left',va='center',fontsize=8.5,color='.35');axs[1,0].set(title='(c) First reversal versus later acceleration',ylabel='Selected date − first reversal (months)')
        for route,col in [('A',BLACK),('D',GRAY)]:
            for mask,ls,marker in [('complete','-','o'),('observed','--','^')]:
                g=summary.loc[summary.route.eq(route)&summary.setting.eq('original')&summary.rule.eq('highest_probability')&summary.regime.eq('median')&summary.weak_slope.eq(.6)&summary['mask'].eq(mask)].sort_values('onset');x=pd.to_datetime(g.onset).dt.year
                axs[1,1].errorbar(x,g.onset_hit_rate,yerr=[g.onset_hit_rate-g.onset_low,g.onset_high-g.onset_hit_rate],c=col,ls=ls,marker=marker,ms=3,capsize=2,label=route+': '+('full' if mask=='complete' else 'gaps'))
        axs[1,1].set(title='(d) Onset hit: within ±12 months',xlabel='Year of first reversal',ylabel='Fraction of all 200 realizations',ylim=(-.03,1.03),xticks=[2016,2018,2019,2020]);axs[1,1].legend(frameon=False,fontsize=8.5,ncol=2)
        self.save(fig,'story_05_detection_scenarios')
        fig,axs=self.subplots(2,2,165,sharex=True,sharey=True)
        for i,(route,col) in enumerate([('A',BLACK),('D',GRAY)]):
            for j,mask in enumerate(['complete','observed']):
                ax=axs[i,j]
                for weak,ls,marker in [(.3,':','s'),(.6,'-','o'),(1.2,'--','^')]:
                    g=summary.loc[summary.route.eq(route)&summary.setting.eq('original')&summary.rule.eq('highest_probability')&summary.regime.eq('median')&summary.weak_slope.eq(weak)&summary['mask'].eq(mask)].sort_values('onset');ax.plot(pd.to_datetime(g.onset).dt.year,g.onset_hit_rate,c=col,ls=ls,marker=marker,label=f'+{weak:g} cm/yr')
                ax.set(title=f'{route}: {mask}',ylim=(-.03,1.03),xticks=[2016,2018,2019,2020],ylabel='Onset-hit fraction',xlabel='First-reversal year');ax.legend(frameon=False,fontsize=8.5)
        self.supplement_save(fig,'monitoring_gap_grid','Complete two-stage median-residual grid. First-reversal and acceleration stages remain 36 months apart. Curves show unconditional first-date accuracy, not merely candidate detection. The 2020 case has only 20 months after acceleration.')
        fig,axs=self.subplots(1,2,95)
        search=self.nr('inference/calibration_search.csv');validation=self.nr('inference/heldout_validation.csv') if (self.new/'inference/heldout_validation.csv').exists() else None
        for ax,route in zip(axs,['A','D']):
            g=search.loc[search.route.eq(route)&search.scenario.ne('sustained_reversal')].groupby(['probability_threshold','gain_threshold']).upper_onesided95.max().reset_index();mat=g.pivot(index='gain_threshold',columns='probability_threshold',values='upper_onesided95');ax.pcolormesh(np.arange(6)-.5,np.arange(5)-.5,mat.to_numpy(),vmin=0,vmax=max(.1,float(search.upper_onesided95.max())),cmap='Greys',shading='flat',edgecolors='none',rasterized=False);ax.set_xlim(-.5,4.5);ax.set_ylim(-.5,3.5);ax.set_xticks(range(5),['.50','.70','.90','.95','.99']);ax.set_yticks(range(4),['.5','1','1.5','2']);ax.set(title=f'Route {route}: worst null upper bound',xlabel='Candidate probability threshold',ylabel='Slope-gain threshold (cm/yr)')
            for y in range(4):
                for x in range(5):ax.text(x,y,f'{mat.iloc[y,x]:.2f}',ha='center',va='center',fontsize=8.5,color='white' if mat.iloc[y,x]>.4 else BLACK)
        self.supplement_save(fig,'monitoring_calibration_grid','Maximum one-sided 95% Wilson upper bound across the six null scenario–residual combinations. A feasible setting must be at or below 0.05 in every null combination; threshold search is restricted to the calibration split.')
    def observations(self):
        source=self.nr('diagnostics/shared_noah/well_series.csv',parse_dates=['date']);self.table('fig06_series',source);correlations=[];fig,axs=self.subplots(2,2,177,gridspec_kw={'height_ratios':[1.1,1]})
        for ax,region in zip(axs[0],['north','south']):
            for aq,ls in [('unconfined','--'),('confined',':')]:
                g=source.loc[source.region.eq(region)&source.aquifer.eq(aq)&source.support.eq('fixed_cells')];ax.plot(g.date,(g.well-g.well.mean())/g.well.std(),c=GRAY,ls=ls,lw=1,label=aq.capitalize())
            for component,ls,label in [('gwa_cm','-','GWA'),('twsa_cm','-.','TWSA')]:
                g=source.loc[source.region.eq(region)&source.aquifer.eq('unconfined')&source.support.eq('fixed_cells')];v=g[component];ax.plot(g.date,(v-v.mean())/v.std(),c=BLACK,ls=ls,lw=1,label=label)
            ax.axvspan(pd.Timestamp('2023-07-01'),pd.Timestamp('2024-01-01'),color='.9');ax.set(title=f'({"a" if region=="north" else "b"}) {region.capitalize()}: matched trajectories',ylabel='Standardized anomaly',ylim=(-2.2,2.7));ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs[0,0].legend(frameon=False,fontsize=8.5,ncol=2)
        for region,col,offset in [('north',BLUE,-.10),('south',RED,.10)]:
            for j,(model,component) in enumerate([('shared_noah','gwa_cm'),('era2','gwa_cm'),('era289','gwa_cm'),('shared_noah','twsa_cm')]):
                d=self.nr(f'diagnostics/{model}/well_correlations.csv');r=d.loc[d.region.eq(region)&d.aquifer.eq('unconfined')&d.support.eq('fixed_cells')&d.version.eq('complete')&d.treatment.eq('raw')&d.component.eq(component)].iloc[0]
                correlations.append(r.to_dict())
                axs[1,0].errorbar(j+offset,r.r,yerr=[[r.r-r.low],[r.high-r.r]],fmt='o',c=col,capsize=2,ms=4,label=region.capitalize() if j==0 else None)
        axs[1,0].set_xticks(range(4),['Noah','ERA\n0–2 m','ERA\n0–2.89 m','TWSA']);axs[1,0].set(ylim=(-1,1),ylabel='Raw well correlation',title='(c) Unconfined: component sensitivity');axs[1,0].axhline(0,c='.6',ls=':');axs[1,0].legend(frameon=False,fontsize=8.5)
        self.table('fig06_component_correlations',pd.DataFrame(correlations))
        # Descriptive cell-level trend correspondence (added comparator; no model fits).
        cell=self.read(ROOT/'outputs/monitoring_revision/2026-10-19-cell-trends/cell_well_trends.csv');cell=cell.loc[cell.aquifer.eq('unconfined')];self.table('fig06_cell_trends',cell)
        for region,col in [('north',BLUE),('south',RED)]:
            g=cell.loc[cell.region_tag.eq(region)]
            axs[1,1].scatter(g.twsa_cm_per_yr,g.well_m_per_yr,c=col,marker='o',s=24,edgecolors='white',linewidths=.4,label=region.capitalize()+' TWSA')
            axs[1,1].scatter(g.gwa_cm_per_yr,g.well_m_per_yr,facecolors='none',edgecolors=col,marker='s',s=20,linewidths=.8,label=region.capitalize()+' GWA')
        axs[1,1].axvline(0,c='.6',ls=':',lw=.8)
        axs[1,1].set(xlabel='Storage trend (cm/yr)',ylabel='Unconfined well trend (m/yr)',title='(d) Cell trends, 2019–2024',ylim=(-.1,2.9));axs[1,1].legend(frameon=False,fontsize=8.5,ncol=2,loc='upper center',handletextpad=.2,columnspacing=.8)
        self.save(fig,'validation_06_observation_support')
    def extra_supplements(self):
        fig,ax=self.subplots(1,1,90);ax.set_axis_off()
        boxes=[(.02,.62,.28,.25,'Fixed inputs and\ncalendar / support audit'),(.37,.62,.27,.25,'Trends, events,\nboundary and LSM'),(.71,.62,.27,.25,'Regional magnitude\nand uncertainty'),(.02,.12,.28,.25,'A / D candidate pools\nrecord-prefix refits'),(.37,.12,.27,.25,'Known truths:\nnulls, reversal,\nacceleration'),(.71,.12,.27,.25,'Held-out rules, real-data\napplication and wells')]
        for x,y,w,h,label in boxes:ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=GRAY,lw=.8));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.5)
        for x1,y1,x2,y2 in [(.30,.75,.37,.75),(.64,.75,.71,.75),(.30,.25,.37,.25),(.64,.25,.71,.25),(.16,.62,.16,.37)]:ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=GRAY,lw=.9))
        self.supplement_save(fig,'monitoring_workflow','Analysis flow. Calibration and held-out validation use disjoint realizations. Real-data reapplication is retrospective; fixed-well dates are observational comparators, not storage-onset truth.')
        example=self.nr('inference/illustrative_simulation.csv',parse_dates=['date']);fig,ax=self.subplots(1,1,85)
        ax.plot(example.date,example.true_trend,c=BLACK,lw=1.3,label='Known trend');ax.plot(example.date,example.observed_input,c=GRAY,lw=.6,label='Seasonal / noisy observed input');ax.scatter(example.date,example.observed_input,c=GRAY,s=3)
        ax.axvspan(pd.Timestamp('2017-07-01'),pd.Timestamp('2018-05-01'),color='.9');ax.set(ylabel='Water-equivalent thickness (cm)',title='Formal realization: 2018 reversal, median residual regime');ax.legend(frameon=False,fontsize=8.5);ax.xaxis.set_major_locator(mdates.YearLocator(4));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        self.supplement_save(fig,'monitoring_noisy_realization','The first formal median-residual realization for the 2018/+0.6 two-stage scenario. The illustration is selected by index and setting, not its fitted outcome; its digest matches the formal input record. Seasonality, AR(1) variability and the actual missing-month mask are retained.')
        p=self.nr('inference/real_selections.csv');slopes=self.nr('diagnostics/archived/cell_trends.csv')
        fig,axs=self.subplots(1,3,90)
        norm=plt.Normalize(-6,6)
        for ax,model,title in zip(axs,['archived','shared_noah','era2'],['Archived GWA','Same-mask Noah','ERA5-Land 0–2 m']):
            data=self.nr(f'diagnostics/{model}/cell_trends.csv');values=data.loc[data.start.eq('2019-01-01')&data.boundary.eq('original')&data.version.eq('complete')&data.method.eq('OLS')].set_index('grid_id').slope
            for gid in self.grids.grid_id:
                _,x,y=gid.split('_');ax.add_patch(Rectangle((float(x),float(y)),1,1,facecolor=plt.get_cmap('BrBG')(norm(values[gid])),edgecolor='white',lw=.3))
            ax.set(xlim=(113,120),ylim=(34.5,40.5),title=title,xticks=[114,118],yticks=[35,39]);ax.set_aspect(1.2)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='BrBG'),ax=list(axs),orientation='horizontal',shrink=.7,pad=.04,label='2019–Aug 2024 OLS trend (cm/yr)')
        self.supplement_save(fig,'monitoring_cell_trends','Cell OLS trends during January 2019–August 2024. Archived residual, same-mask Noah and ERA5-Land 0–2 m separate support and component sensitivities. These polygons are aggregation units, not independent GRACE footprints; no naive cell significance is claimed.')
        fig,axs=self.subplots(2,3,140)
        for i,route in enumerate(['A','D']):
            slope=slopes.loc[slopes.start.eq('2019-01-01')&slopes.boundary.eq('original')&slopes.version.eq('complete')&slopes.method.eq('OLS')].set_index('grid_id').slope
            a=p.loc[p.route.eq(route)&p.rule.eq('highest_probability')&p.setting.eq('original')&p.endpoint.eq('2023-12-01')].set_index('grid_id');b=p.loc[p.route.eq(route)&p.rule.eq('highest_probability')&p.setting.eq('original')&p.endpoint.eq('2024-08-01')].set_index('grid_id')
            aa=pd.to_datetime(a.selected_date);bb=pd.to_datetime(b.selected_date);av=aa.dt.year+(aa.dt.month-1)/12;bv=bb.dt.year+(bb.dt.month-1)/12;revision=(bb.dt.year-aa.dt.year)*12+bb.dt.month-aa.dt.month
            for j,(values,title,cmap,vmin,vmax) in enumerate([(av,'Dec 2023','cividis',2015,2024),(bv,'Aug 2024','cividis',2015,2024),(revision,'Revision','PuOr',-72,72)]):
                ax=axs[i,j];norm=plt.Normalize(vmin,vmax)
                for gid in self.grids.grid_id:
                    _,x,y=gid.split('_');value=values.get(gid,np.nan);ax.add_patch(Rectangle((float(x),float(y)),1,1,facecolor=plt.get_cmap(cmap)(norm(value)) if np.isfinite(value) else '.85',edgecolor='white',lw=.3))
                ax.set(xlim=(113,120),ylim=(34.5,40.5),title=f'{route}: {title}');ax.set_aspect(1.2);ax.set_xticks([114,118]);ax.set_yticks([35,39])
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(2015,2024),cmap='cividis'),ax=list(axs[:,:2].flat),orientation='horizontal',shrink=.8,pad=.04,label='Selected year')
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-72,72),cmap='PuOr'),ax=list(axs[:,2].flat),orientation='horizontal',shrink=.9,pad=.04,label='Revision (months)')
        self.supplement_save(fig,'monitoring_cell_maps','Cell maps of original-rule selected dates and signed revisions. Gray means no qualifying date or unavailable revision. One-degree polygons are aggregation units and do not indicate independent satellite resolution; no naive pixel significance is displayed.')
        fig,axs=self.subplots(1,2,100)
        climate=self.nr('diagnostics/regional_precipitation.csv',parse_dates=['date'])
        for ax,region,col in zip(axs,['north','south'],[BLUE,RED]):
            g=climate.loc[climate.region_tag.eq(region)&climate.date.ge('2019-01-01')];ax.bar(g.date,g.precip_anomaly_mm,width=23,color=col,alpha=.65);ax.axhline(0,c='.6',lw=.6);ax.set(title=region.capitalize(),ylabel='ERA5-Land precipitation anomaly (mm)');ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        self.supplement_save(fig,'monitoring_precipitation','Monthly ERA5-Land precipitation anomalies relative to the 2003–2014 calendar-month climatology. Regional weights use lowland coverage and latitude; event windows are sensitivity conventions, not estimated causal response durations.')
        fig,axs=self.subplots(1,2,105)
        d=self.nr('diagnostics/archived/trend_contrasts.csv');d=d.loc[d.boundary.eq('original')&d.version.eq('complete')]
        for ax,start in zip(axs,['2019-01-01','2022-01-01']):
            rows=[]
            for y,(method,size) in enumerate([('OLS',2),('OLS',3),('Theil-Sen',2),('Theil-Sen',3)]):
                r=d.loc[d.start.eq(start)&d.method.eq(method)&d.block_degrees.eq(size)].iloc[0];ax.errorbar(r.estimate,y,xerr=[[r.estimate-r.low],[r.high-r.estimate]],fmt='o' if method=='OLS' else '^',c=BLACK if method=='OLS' else GRAY,capsize=2);rows.append(method+f' {size}°')
            ax.set_yticks(range(4),rows);ax.invert_yaxis();ax.axvline(0,c='.6',ls=':');ax.set(title=start[:4]+'–Aug 2024',xlabel='South − north (cm/yr)')
        self.supplement_save(fig,'monitoring_estimators','Estimator and block-size sensitivity under the newly declared resampling. OLS intervals jointly resample time and space; Theil–Sen intervals resample spatial blocks conditional on fitted slopes. Their interval definitions are deliberately distinguished.')
        fig,axs=self.subplots(1,2,105)
        d=self.read(VALIDATION/'uniform/well_multiscale_correlations.csv')
        for region,col,offset in [('north',BLUE,-.08),('south',RED,.08)]:
            g=d.loc[d.support.eq('common18')&d.weighting.eq('equal_cell')&d.aquifer.eq('unconfined')&d.component.eq('gwa_cm')&d.treatment.eq('difference')].copy();g=g.loc[g.region.eq(region)].sort_values('aggregation_months')
            axs[0].errorbar(np.arange(4)+offset,g.r,yerr=[g.r-g.low,g.high-g.r],fmt='o',c=col,capsize=2,label=region.capitalize());axs[0].set_xticks(range(4),['1\n(n=67)','3\n(n=21)','6\n(n=10)','12\n(n=4)']);axs[0].set(ylim=(-1,1),xlabel='Aggregation months',ylabel='Change correlation');axs[0].axhline(0,c='.6',ls=':')
            lag=self.read(VALIDATION/'uniform/well_lags_0_12.csv');q=lag.loc[lag.support.eq('common18')&lag.weighting.eq('equal_cell')&lag.aquifer.eq('unconfined')&lag.component.eq('gwa_cm')&lag.region.eq(region)];axs[1].plot(q.lag_months,q.r,c=col,marker='o',ms=3,label=region.capitalize())
        axs[0].legend(frameon=False);axs[1].set(xlabel='GWA leads wells (months)',ylabel='Residual correlation');axs[1].axhline(0,c='.6',ls=':')
        self.supplement_save(fig,'monitoring_timescales','Retained uniform-support monthly-to-annual and 0–12-month lag diagnostics. Annual changes have only four pairs and are descriptive. Points are not connected across aggregation scales; no preferred lag is selected after inspecting correlations.')
    def graphical(self):
        fig,axs=plt.subplots(1,3,figsize=(240*old.MM,90*old.MM),layout='constrained')
        d=self.nr('diagnostics/archived/trend_contrasts.csv');q=d.loc[d.start.eq('2019-01-01')&d.boundary.eq('original')&d.method.eq('OLS')&d.block_degrees.eq(3)&d.version.isin(['complete','exclude_both'])]
        axs[0].errorbar(q.estimate,range(len(q)),xerr=[q.estimate-q.low,q.high-q.estimate],fmt='o',c=BLACK,capsize=3);axs[0].axvline(0,c='.6',ls=':');axs[0].set_yticks(range(len(q)),['Complete','Event windows excluded']);axs[0].set(xlabel='South − north (cm/yr)',title='Magnitude and uncertainty')
        axs[1].bar(['A','D'],[13/24,13/23],color=[BLACK,GRAY]);axs[1].set(ylim=(0,1),ylabel='Within six months',title='Dates revise as records extend');axs[1].text(.5,.85,'13/24 and 13/23 dates stable',transform=axs[1].transAxes,ha='center',fontsize=9.5)
        axs[2].set_axis_off();axs[2].text(.02,.92,'Report together',va='top',fontsize=14,fontweight='bold');axs[2].text(.02,.70,'Trend and spatial scale\nDate rule and real endpoint\nRevision history and coverage\nMatched observation support',va='top',fontsize=11,linespacing=1.8)
        self.save(fig,'graphical_abstract_1328x531')
    def run(self,figure01_only=False,figure02_only=False):
        if figure02_only:
            self.trends()
        else:
            self.wells()
        if not figure01_only and not figure02_only:
            self.trends();self.endpoints();self.processing();self.simulation();self.observations();self.extra_supplements();self.graphical()
        self.inputs[__file__.replace('\\','/').split(str(ROOT).replace('\\','/')+'/')[-1]]=sha(__file__)
        json_write(self.out/'supplement_figures.json',self.supplement)
        ledger=dict(render_only=True,model_fits_run=0,figure01_context_only=figure01_only,figure02_only=figure02_only,inputs=self.inputs,primary_endpoint='2024-08-01',files={p.relative_to(self.out).as_posix():sha(p) for name in ['figures','tables'] for p in (self.out/name).iterdir()},min_font_pt=8.5)
        json_write(self.out/'figure_evidence_ledger.json',ledger)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
    scope=p.add_mutually_exclusive_group();scope.add_argument('--figure01-only',action='store_true');scope.add_argument('--figure02-only',action='store_true')
    a=p.parse_args();Figures(a.output_dir).run(figure01_only=a.figure01_only,figure02_only=a.figure02_only)

if __name__=='__main__':main()
