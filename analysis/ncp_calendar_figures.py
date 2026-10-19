"""Render current manuscript figures from calendar-revision evidence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED, BLACK = "#286090", "#b33e36", "#222222"
COLORS = {"all": BLACK, "north": BLUE, "south": RED}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    evidence, out = Path(args.evidence), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 280})
    b = pd.read_csv(evidence / "breakpoints_calendar_stl.csv", parse_dates=["onset_date"])
    panel = pd.read_csv(ROOT / "data/raw/grid_monthly_panel.csv", parse_dates=["date"])
    def save(fig, name):
        fig.savefig(out / name, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.axis("off")
    boxes = [(0.01,.61,.29,.30,"Monthly storage proxy\nGRACE minus land-water components\n27 cells; 261 calendar months\n218 observed months per cell"),(.355,.61,.29,.30,"Calendar-preserving detection\nSTL seasonal component\nBEAST + single-break regression\nMissing months retained"),(.70,.61,.29,.30,"Recovery reporting\nCandidate from January 2015\nPositive slope; gain > 0.5 cm/yr\nProbability / BIC thresholds"),(.01,.08,.29,.32,"Spatial uncertainty\nSouth minus north\n1°, 2° and 3° spatial blocks\nAll breaks and 2018+ subset"),(.355,.08,.29,.32,"Independent observations\nShallow wells, grid-first means\nRaw / detrended / differenced r\nStable 2018+ well cohort"),(.70,.08,.29,.32,"Sensitivity experiments\nObserved-only seasonal model\nWater-fraction residualization\nRecord truncated at 2020")]
    for x,y,w,h,txt in boxes:
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=BLUE,lw=1.2))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",linespacing=1.6,fontsize=10)
    for x in [.315,.66]:
        ax.annotate("",xy=(x+.035,.76),xytext=(x,.76),arrowprops=dict(arrowstyle="->",color=BLACK))
    for x in [.15,.5,.845]:
        ax.annotate("",xy=(x,.42),xytext=(x,.58),arrowprops=dict(arrowstyle="->",color=BLACK))
    save(fig,"figure_02_workflow.png")

    fig, ax = plt.subplots(figsize=(8.7,4.3))
    for region, data in [("all",panel),("north",panel[panel.region_tag.eq("north")]),("south",panel[panel.region_tag.eq("south")])]:
        s=data.groupby("date").gwa_cm.mean()
        ax.plot(s.index,s,label={"all":"All NCP","north":"North","south":"South"}[region],c=COLORS[region],lw=1.1)
    ax.axvspan(pd.Timestamp("2017-06-01"),pd.Timestamp("2018-05-01"),color="0.9",zorder=-2)
    ax.text(pd.Timestamp("2017-12-01"),12,"GRACE gap",rotation=90,ha="center",va="top",fontsize=8)
    ax.axvline(pd.Timestamp("2014-12-12"),c="0.5",ls=":")
    ax.text(pd.Timestamp("2014-10-01"),12,"SNWD",rotation=90,ha="right",va="top",fontsize=8)
    ax.set(ylabel="GWA (cm water-equivalent thickness)",xlabel="Year")
    ax.legend(ncol=3,frameon=False,loc="lower left")
    save(fig,"figure_03_regional_gwa_timeseries.png")

    valid=b.dropna(subset=["onset_date"])
    eff=pd.read_csv(evidence/"spatial_effects.csv").query("block_degrees == 2")
    fig, axes=plt.subplots(1,2,figsize=(9,4.3))
    rng=np.random.default_rng(42)
    for i,r in enumerate(["north","south"]):
        sub=valid.loc[valid.region_tag.eq(r)]
        axes[0].scatter(sub.onset_date,i+rng.uniform(-.1,.1,len(sub)),color=COLORS[r],s=24)
        axes[1].scatter(np.full(len(sub),i)+rng.uniform(-.1,.1,len(sub)),sub.post_trend_cm_per_year,color=COLORS[r],s=24)
        axes[1].plot([i-.22,i+.22],[sub.post_trend_cm_per_year.mean()]*2,c=BLACK,lw=1.4)
    axes[0].set(yticks=[0,1],yticklabels=["North (n=8)","South (n=16)"],ylim=(-.5,1.5),title="(a) Selected recovery-related breaks",xlabel="Year")
    axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set(xticks=[0,1],xticklabels=["North","South"],ylabel="Post-break trend (cm/yr)",title="(b) Observed-month trends")
    for ax,row,unit in zip(axes,eff.itertuples(),["months","cm/yr"]):
        ax.text(.02,-.25,f"South − North: {row.estimate:.2f} {unit}\n95% block interval [{row.ci_lower:.2f}, {row.ci_upper:.2f}]",transform=ax.transAxes,fontsize=9)
    fig.subplots_adjust(wspace=.36,bottom=.25)
    save(fig,"figure_04_recovery_metrics.png")

    fig,ax=plt.subplots(figsize=(6.7,6))
    cmap=plt.get_cmap("Blues")
    norm=plt.Normalize(2015,2022)
    for row in b.itertuples():
        lon,lat=map(float,row.grid_id.split("_")[1:])
        value=row.onset_date.year+(row.onset_date.month-1)/12 if pd.notna(row.onset_date) else np.nan
        ax.add_patch(Rectangle((lon,lat),1,1,facecolor=cmap(norm(value)) if np.isfinite(value) else "0.85",edgecolor="white",lw=.7,hatch=None if np.isfinite(value) else "//"))
    ax.axhline(37.5,c=RED,ls="--",lw=1)
    ax.text(112.2,37.58,"North / South",color=RED,fontsize=9)
    ax.set(xlim=(112,120.1),ylim=(34.3,40.7),xlabel="Longitude (°E)",ylabel="Latitude (°N)")
    ax.set_aspect(1/np.cos(np.deg2rad(37)))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.04,pad=.04)
    cb.set_label("Selected break (year)")
    ax.legend(handles=[Patch(facecolor=".85",hatch="//",label="No reportable recovery break")],loc="lower right",fontsize=8,framealpha=.95)
    save(fig,"figure_05_trajectory_map.png")

    water=pd.read_csv(evidence/"water_association.csv")
    sens=pd.read_csv(evidence/"sensitivity_summary.csv").iloc[:4]
    fig,axes=plt.subplots(1,2,figsize=(9,4.1))
    for r in ["north","south"]:
        sub=water[water.region_tag.eq(r)]
        axes[0].scatter(sub.water_change,sub.post_trend_cm_per_year,c=COLORS[r],label=r.capitalize(),s=30)
    axes[0].set(xlabel="Surface-water fraction change ΔW",ylabel="Post-break trend (cm/yr)",title="(a) Grid-level correspondence")
    axes[0].legend(frameon=False)
    labels=["Calendar STL","Observed harmonics","Water: current","Water: lags 0–3"]
    axes[1].scatter(sens.onset_south_minus_north,np.arange(4),c=BLUE,s=35)
    axes[1].set(yticks=np.arange(4),yticklabels=labels,xlabel="South − North break timing (months)",title="(b) Proxy and seasonality choices")
    axes[1].axvline(0,c="0.5",ls=":")
    axes[1].invert_yaxis()
    fig.tight_layout(w_pad=1.8)
    save(fig,"figure_06_surface_water.png")

    case=pd.read_csv(ROOT/"data/raw/hutuo_case_buffers_raw.csv",parse_dates=["date"])
    fig,axes=plt.subplots(2,1,figsize=(8.7,6),sharex=True)
    for i,(buffer,sub) in enumerate(case.groupby("buffer_km",sort=False)):
        axes[0].plot(sub.date,sub.water_frac,label=buffer.replace("_","–")+" km",color=[BLACK,BLUE,RED,".6"][i],lw=1)
    axes[0].set(ylabel="Water fraction",title="(a) Hutuo corridor buffers")
    axes[0].legend(ncol=4,frameon=False,fontsize=8)
    for i,gid in enumerate(case.case_grid_id.unique()):
        sub=panel[panel.grid_id.eq(gid)&panel.date.ge(case.date.min())]
        axes[1].plot(sub.date,sub.gwa_cm,label=gid,color=[BLACK,BLUE][i])
    axes[1].set(ylabel="GWA (cm)",xlabel="Year",title="(b) Observed GWA in representative cells")
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    save(fig,"figure_08_hutuo_case_response.png")

    series=pd.read_csv(evidence/"validation_series.csv",parse_dates=["date"])
    metrics=pd.read_csv(evidence/"validation_metrics.csv")
    stable=pd.read_csv(evidence/"stable_wells/validation_metrics.csv")
    fig,axes=plt.subplots(3,1,figsize=(8.7,7.5),sharex=True)
    for ax,r,letter in zip(axes,["all","north","south"],"abc"):
        sub=series[series.region_tag.eq(r)].copy()
        for col,label,c in [("gwa_cm","Remote GWA",BLACK),("well_anomaly","Shallow wells",BLUE)]:
            z=(sub[col]-sub[col].mean())/sub[col].std()
            ax.plot(sub.date,z,c=c,lw=1,label=label)
        ax.axvspan(pd.Timestamp("2019-01-01"),sub.date.max(),color=".94",zorder=-1)
        row=metrics.query("region == @r and period == '2019+'").iloc[0]
        s=stable.query("region == @r and period == '2019+'").iloc[0]
        ax.set(ylabel="Standardized anomaly",title=f"({letter}) {'All NCP' if r=='all' else r.capitalize()} | post-2019 r = {row.pearson:.2f}; stable cohort r = {s.pearson:.2f}")
    axes[0].legend(ncol=2,frameon=False,loc="lower left",fontsize=9)
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    save(fig,"figure_09_validation.png")

    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    for ax in axes: ax.axis("off")
    texts=[("OBSERVATIONS","27 North China Plain cells\n2003–2024 monthly storage proxy\nCalendar gaps retained\nPublic shallow-well records"),("RECOVERY PATTERN","24 recovery-related breaks\nMedian: March 2020\nInterquartile span: Aug 2019–Aug 2021\nTiming order differs between models"),("CROSS-CHECKS","Higher southern post-break slopes\nSpatial-scale-sensitive uncertainty\nWeak northern raw well correlation\nRecovery is sensitive to record length")]
    for ax,(title,body) in zip(axes,texts):
        ax.text(.02,.85,title,c=BLUE,fontweight="bold",fontsize=14)
        ax.text(.02,.67,body,va="top",fontsize=11,linespacing=2.)
    fig.tight_layout()
    save(fig,"graphical_abstract_1328x531.png")


if __name__ == "__main__":
    main()
