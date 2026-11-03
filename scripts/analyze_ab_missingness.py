"""Compare raw and history-available missingness using only fixed A+B training patients."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/.matplotlib"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from threadpoolctl import threadpool_limits
from sepsis_warning.ab_regularization import OUTPUT,PROCESSED,REPORT
from sepsis_warning.ab_expansion import write_json,content_hash
from sepsis_warning.data import COLUMNS,FEATURE_NAMES
from sepsis_warning.missingness import patient_means,event_standardized


def main():
    out=OUTPUT/"missingness";out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OUTPUT/"protocol.json").read_text())
    rows=pd.read_csv(PROCESSED/"training_rows.csv.gz")
    patients=pd.read_csv(OUTPUT/"patient_cv_folds.csv")
    raw=np.load(PROCESSED/"training_features.npy",mmap_mode="r")
    assert len(rows)==len(raw) and len(patients)==protocol["training_patients"]
    codes=pd.Categorical(rows.patient_id,categories=patients.patient_id).codes
    assert (codes>=0).all()
    variables=COLUMNS[:40]
    raw_missing=raw[:,[FEATURE_NAMES.index(name+"__missing_now") for name in variables]].astype(bool)
    available_columns=[FEATURE_NAMES.index(name+("__latest" if i<34 else "__current")) for i,name in enumerate(variables)]
    latest_missing=np.isnan(raw[:,available_columns])
    raw_patient=patient_means(raw_missing,codes,len(patients))
    latest_patient=patient_means(latest_missing,codes,len(patients))
    first24=rows.hour.le(24).to_numpy()
    first_patient=patient_means(raw_missing,codes,len(patients),first24)
    target_prevalence=float(patients.has_event.mean())
    patient_hospital=patients.hospital.to_numpy();hour_hospital=rows.hospital.to_numpy()
    event=patients.has_event.to_numpy(bool)
    summary,strata,bands,burden=[],[],[],[]
    for hospital in ["A","B"]:
        pmask=patient_hospital==hospital;hmask=hour_hospital==hospital
        standardized=event_standardized(raw_patient[pmask],event[pmask],target_prevalence)
        for j,variable in enumerate(variables):
            summary.append({"hospital":hospital,"variable":variable,"patients":int(pmask.sum()),"hours":int(hmask.sum()),
                "raw_hour_missing_rate":float(raw_missing[hmask,j].mean()),"latest_hour_missing_rate":float(latest_missing[hmask,j].mean()),
                "raw_patient_mean_missing_rate":float(raw_patient[pmask,j].mean()),"latest_patient_mean_missing_rate":float(latest_patient[pmask,j].mean()),
                "patients_never_observed_rate":float((raw_patient[pmask,j]==1).mean()),
                "raw_event_standardized_patient_missing_rate":float(standardized[j]),
                "raw_first24_patient_mean_missing_rate":float(np.nanmean(first_patient[pmask,j])),
                "first24_patients":int(np.isfinite(first_patient[pmask,j]).sum())})
        for status in [False,True]:
            pgroup=pmask&(event==status);hgroup=hmask&(event[codes]==status)
            for j,variable in enumerate(variables):
                strata.append({"hospital":hospital,"has_event":status,"variable":variable,"patients":int(pgroup.sum()),"hours":int(hgroup.sum()),
                    "raw_hour_missing_rate":float(raw_missing[hgroup,j].mean()),"latest_hour_missing_rate":float(latest_missing[hgroup,j].mean()),
                    "raw_patient_mean_missing_rate":float(raw_patient[pgroup,j].mean())})
        for name,mask in [("1-6",rows.hour.le(6)),("7-12",rows.hour.between(7,12)),("13-24",rows.hour.between(13,24)),("25+",rows.hour.gt(24))]:
            select=hmask&mask.to_numpy()
            for j,variable in enumerate(variables):
                bands.append({"hospital":hospital,"hour_band":name,"variable":variable,"hours":int(select.sum()),
                    "patients":int(rows.loc[select,"patient_id"].nunique()),"raw_hour_missing_rate":float(raw_missing[select,j].mean()),
                    "latest_hour_missing_rate":float(latest_missing[select,j].mean())})
        burden.append({"hospital":hospital,"patients":int(pmask.sum()),"event_patients":int((pmask&event).sum()),
            "event_patient_prevalence":float(event[pmask].mean()),"median_eligible_hours":float(np.median(np.bincount(codes)[pmask])),
            "median_patient_raw_missing_fraction":float(np.median(raw_patient[pmask].mean(axis=1))),
            "mean_hour_raw_missing_fraction":float(raw_missing[hmask].mean()),"mean_hour_latest_missing_fraction":float(latest_missing[hmask].mean()),
            "hours_without_any_lab_measurement_rate":float(raw_missing[hmask,8:34].all(axis=1).mean())})
    summary=pd.DataFrame(summary);strata=pd.DataFrame(strata);bands=pd.DataFrame(bands)
    summary.to_csv(out/"by_hospital.csv",index=False);strata.to_csv(out/"by_event.csv",index=False)
    bands.to_csv(out/"by_hour_band.csv",index=False);pd.DataFrame(burden).to_csv(out/"patient_composition.csv",index=False)
    columns=["raw_hour_missing_rate","latest_hour_missing_rate","raw_patient_mean_missing_rate","latest_patient_mean_missing_rate",
        "patients_never_observed_rate","raw_event_standardized_patient_missing_rate","raw_first24_patient_mean_missing_rate"]
    a=summary.loc[summary.hospital.eq("A")].set_index("variable");b=summary.loc[summary.hospital.eq("B")].set_index("variable")
    comparisons=[]
    for variable in variables:
        record={"variable":variable}
        for col in columns:
            record[f"A_{col}"]=float(a.loc[variable,col]);record[f"B_{col}"]=float(b.loc[variable,col])
            record[f"B_minus_A_{col}"]=float(b.loc[variable,col]-a.loc[variable,col])
        comparisons.append(record)
    comparison=pd.DataFrame(comparisons);comparison.to_csv(out/"cross_hospital.csv",index=False)
    # Diagnostic only: whole eligible-episode missing fractions never become sepsis inputs.
    label=patients.hospital.eq("B").to_numpy(np.int8)
    hospital_p=np.full(len(patients),np.nan);diagnostics=[]
    for fold in range(3):
        held=patients.fold.eq(fold).to_numpy();training=~held
        scaler=StandardScaler().fit(raw_patient[training])
        classifier=LogisticRegression(C=1,solver="liblinear",max_iter=2000,random_state=42)
        with threadpool_limits(limits=1):classifier.fit(scaler.transform(raw_patient[training]),label[training])
        hospital_p[held]=classifier.predict_proba(scaler.transform(raw_patient[held]))[:,1]
        diagnostics.append({"fold":fold,"patients":int(held.sum()),"hospital_auroc":float(roc_auc_score(label[held],hospital_p[held])),
            "balanced_accuracy":float(balanced_accuracy_score(label[held],hospital_p[held]>=.5))})
    pd.DataFrame(diagnostics).to_csv(out/"hospital_classifier_folds.csv",index=False)
    pd.DataFrame({"patient_id":patients.patient_id,"fold":patients.fold,"hospital":patients.hospital,"probability_B":hospital_p}).to_csv(out/"hospital_classifier_oof.csv",index=False)
    diagnostic={"pooled_oof_hospital_auroc":float(roc_auc_score(label,hospital_p)),"features":"40 whole-eligible-episode raw missing fractions per training patient",
        "folds":"Same hospital/event-stratified training-patient folds; scaler fitted within each fold", "sepsis_model_input":False,
        "interpretation":"Describes separability of data sources; does not identify causal effects or documentation errors."}
    write_json(out/"hospital_classifier.json",diagnostic)
    make_plots(comparison,bands,out)
    write_json(out/"manifest.json",{"training_patients":len(patients),"training_hours":len(rows),"variables":40,"test_sets_used":[],
        "target_event_patient_prevalence":target_prevalence,"protocol_sha256":content_hash(OUTPUT/"protocol.json"),
        "input_training_features_sha256":content_hash(PROCESSED/"training_features.npy"),
        "source_fingerprints":{str(p.relative_to(ROOT)):content_hash(p) for p in [Path(__file__),ROOT/"src/sepsis_warning/missingness.py"]}})
    print(comparison.sort_values("B_minus_A_raw_hour_missing_rate",key=abs,ascending=False).head(8)[
        ["variable","A_raw_hour_missing_rate","B_raw_hour_missing_rate","A_latest_hour_missing_rate","B_latest_hour_missing_rate"]].to_string(index=False),flush=True)
    print("Hospital separability from missingness AUROC:",diagnostic["pooled_oof_hospital_auroc"],flush=True)


def make_plots(comparison,bands,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    for ax,metric,title in zip(axes,["raw_hour_missing_rate","latest_hour_missing_rate"],["No observation in current hourly bin","Latest value unavailable (6-hour expiry for physiology)"]):
        part=comparison.sort_values("B_minus_A_"+metric,key=abs,ascending=False).head(12).iloc[::-1]
        y=np.arange(len(part))
        ax.barh(y-.18,part["A_"+metric]*100,height=.34,label="A",color="#367a95")
        ax.barh(y+.18,part["B_"+metric]*100,height=.34,label="B",color="#c27543")
        ax.set(yticks=y,yticklabels=part.variable,xlim=(0,102),xlabel="Missing eligible training hours (%)",title=title)
    fig.suptitle("Training-only missingness: 12 largest absolute hospital gaps per panel")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/"raw_and_latest.png",dpi=180);plt.close(fig)
    part=comparison.sort_values("B_minus_A_raw_patient_mean_missing_rate",key=abs,ascending=False).head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,6.4));y=np.arange(len(part))
    for offset,metric,label,color in [(-.2,"raw_patient_mean_missing_rate","Patient mean, full eligible episode","#367a95"),
        (0,"raw_event_standardized_patient_missing_rate","Common event-patient proportion","#c27543"),
        (.2,"raw_first24_patient_mean_missing_rate","Patient mean, first 24 hours","#648a55")]:
        ax.scatter(part["B_minus_A_"+metric]*100,y+offset,label=label,color=color,s=28)
    ax.axvline(0,color="gray",linestyle="--");ax.set(yticks=y,yticklabels=part.variable,xlabel="B minus A raw missingness (percentage points)",title="Patient weighting and simple case-mix / time controls")
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.005),ncol=1,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/"patient_weighted_controls.png",dpi=180);plt.close(fig)
    order=["1-6","7-12","13-24","25+"]
    pivot=bands.pivot(index="variable",columns=["hospital","hour_band"],values="latest_hour_missing_rate")
    delta=np.column_stack([(pivot["B",band]-pivot["A",band]).reindex(COLUMNS[:40]).to_numpy() for band in order])*100
    fig,ax=plt.subplots(figsize=(8,11));limit=max(10,float(np.nanmax(np.abs(delta))))
    im=ax.imshow(delta,aspect="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
    ax.set(xticks=np.arange(4),xticklabels=order,yticks=np.arange(40),yticklabels=COLUMNS[:40],xlabel="ICU hour band",title="B minus A latest-value missingness, training only")
    ax.tick_params(axis="y",labelsize=8);fig.colorbar(im,ax=ax,label="Percentage points")
    fig.tight_layout();fig.savefig(out/"hour_band_gaps.png",dpi=180);plt.close(fig)


if __name__=="__main__":main()
