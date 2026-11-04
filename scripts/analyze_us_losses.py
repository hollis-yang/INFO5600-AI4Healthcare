"""Read-only, non-ensemble audit of official utility regret and alarm timing.

Training OOF alarms retain each fold's independently selected inner-calibration
threshold. Test alarms retain the already frozen validation threshold. No new
threshold search, fitting, or test-driven policy selection is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "processed_data/friend_563_comparison"
OUT = ROOT / "outputs/challenge_improvement"
REPORT = ROOT / "reports/us_loss_analysis"
PRIMARY = "lgb_bce_lag_observed_leaf63"
MODELS = (PRIMARY, "lgb_bce_base", "lgb_bce_all", "xgb_lag_observed_depth5")
BUCKETS = ("nonseptic_false_alarm", "septic_alarm_before_minus12",
           "missed_minus11_to_minus7", "missed_minus6_to_minus1",
           "missed_onset_to_plus3")
CHINESE = {"nonseptic_false_alarm": "非脓毒症患者误报",
           "septic_alarm_before_minus12": "发病前超过12小时误报",
           "missed_minus11_to_minus7": "−11至−7小时有益窗口未报警",
           "missed_minus6_to_minus1": "−6至−1小时未报警",
           "missed_onset_to_plus3": "发病至+3小时未报警"}


def available(path):
    path = Path(path)
    if not path.is_file() or getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise OSError(f"Missing or cloud-evicted input: {path}")
    return path


def read_json(path):
    return json.loads(available(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with available(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def official_function():
    folder = ROOT / "third_party/physionet2019"
    metadata = read_json(folder / "UPSTREAM.json")
    assert metadata["commit"] == "467c49b514542be7a4a0bafe40fa2c3b064dda2e"
    for name, entry in metadata["files"].items():
        assert sha(folder / name) == entry["sha256"]
    spec = importlib.util.spec_from_file_location("us_loss_official", folder / "evaluate_sepsis_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_prediction_utility


def delta_integer(labels):
    """Exact marginal utilities in units of 1/180; independent of model code."""
    values = np.full(len(labels), -9, dtype=np.int64)
    positive = np.flatnonzero(labels)
    relative = None
    if len(positive):
        relative = np.arange(len(labels)) - (int(positive[0]) + 6)
        early = (relative >= -12) & (relative <= -6)
        late = (relative > -6) & (relative <= 3)
        values[early] = 30 * (relative[early] + 12)
        values[late] = 20 * (relative[late] + 15)
        values[relative > 3] = 0
    return values, relative


def true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    return np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)


def row_indices(patients):
    return np.concatenate([np.arange(int(p.start), int(p.stop)) for p in patients.itertuples(index=False)])


def load_predictions(model, split, provenance):
    patients_path = BASE / f"{split}_patients.csv"
    labels_path = BASE / f"{split}_labels.npy"
    patients = pd.read_csv(available(patients_path))
    labels = np.load(available(labels_path))
    provenance[str(patients_path.relative_to(ROOT))] = sha(patients_path)
    provenance[str(labels_path.relative_to(ROOT))] = sha(labels_path)
    assert len(labels) == int(patients.hours.sum()) and patients.patient_id.is_unique
    if split == "train":
        scores = np.full(len(labels), np.nan)
        thresholds = np.full(len(labels), np.nan)
        for fold in range(3):
            directory = OUT / "cv" / f"fold{fold}" / model
            record = read_json(directory / "results.json")
            original = patients.loc[patients.cv_fold.eq(fold)]
            local = pd.read_csv(available(directory / "held_patients.csv"))
            np.testing.assert_array_equal(original.patient_id.to_numpy(), local.patient_id.to_numpy())
            indices = row_indices(original)
            path = directory / "held_score.npy"
            values = np.load(available(path))
            assert len(values) == len(indices)
            scores[indices] = values
            thresholds[indices] = record["calibration"]["threshold"]
            provenance[str(path.relative_to(ROOT))] = sha(path)
            provenance[str((directory / "results.json").relative_to(ROOT))] = sha(directory / "results.json")
        expected = pd.read_csv(available(ROOT / "reports/challenge_improvement/cv_metrics.csv"))
        expected_us = float(expected.set_index("candidate").loc[model, "utility_score"])
    else:
        directory = OUT / "evaluation" / split / model
        record = read_json(directory / "metrics.json")
        path = directory / "score.npy"
        scores = np.load(available(path))
        thresholds = np.full(len(labels), record["threshold"])
        expected_us = float(record["utility_score"])
        provenance[str(path.relative_to(ROOT))] = sha(path)
        provenance[str((directory / "metrics.json").relative_to(ROOT))] = sha(directory / "metrics.json")
    assert np.isfinite(scores).all() and np.isfinite(thresholds).all()
    return patients, labels, scores, thresholds, expected_us


def audit(model, split, official, provenance):
    patients, labels, scores, thresholds, expected_us = load_predictions(model, split, provenance)
    person_rows, run_rows, curve_rows, score_rows = [], [], [], []
    official_observed = official_best = official_inactive = 0.
    checked = 0
    for person in patients.itertuples(index=False):
        a, b = int(person.start), int(person.stop)
        y, probability, cutoff = labels[a:b], scores[a:b], thresholds[a:b]
        alarm = probability >= cutoff
        delta, relative = delta_integer(y)
        regret = np.maximum(delta, 0) - delta * alarm
        denominator = int(np.maximum(delta, 0).sum())
        if relative is None:
            masks = {BUCKETS[0]: np.ones(len(y), dtype=bool)}
        else:
            masks = {BUCKETS[1]: relative < -12,
                     BUCKETS[2]: (relative >= -11) & (relative <= -7),
                     BUCKETS[3]: (relative >= -6) & (relative <= -1),
                     BUCKETS[4]: (relative >= 0) & (relative <= 3)}
        partitioned = np.zeros(len(y), bool)
        record = {"model": model, "split": split, "patient_id": person.patient_id,
                  "hospital": person.hospital, "has_event": int(relative is not None),
                  "hours": len(y), "alarm_hours": int(alarm.sum()), "denominator_integer": denominator,
                  "gain_integer": int((delta * alarm).sum()), "regret_integer": int(regret.sum())}
        for category in BUCKETS:
            mask = masks.get(category, np.zeros(len(y), bool))
            partitioned |= mask
            record[category+"_loss_integer"] = int(regret[mask].sum())
            record[category+"_opportunity_hours"] = int(mask.sum())
            record[category+"_loss_hours"] = int((mask & (regret > 0)).sum())
        assert int(regret[~partitioned].sum()) == 0
        assert sum(record[x+"_loss_integer"] for x in BUCKETS) == int(regret.sum())
        record["useful_missed_loss_integer"] = sum(record[x+"_loss_integer"] for x in BUCKETS[2:])
        if model == PRIMARY:
            # Directly call the pinned official scorer for every primary-model
            # patient, instead of validating a formula against itself.
            best = (delta > 0).astype(np.int8)
            inactive = np.zeros(len(y), np.int8)
            u_observed = float(official(y, alarm))
            u_best = float(official(y, best))
            u_inactive = float(official(y, inactive))
            np.testing.assert_allclose((u_best-u_observed)*180, regret.sum(), rtol=0, atol=1e-8)
            np.testing.assert_allclose((u_best-u_inactive)*180, denominator, rtol=0, atol=1e-8)
            official_observed += u_observed; official_best += u_best; official_inactive += u_inactive
            checked += 1
        if relative is not None:
            hits = np.flatnonzero(alarm)
            first = int(relative[hits[0]]) if len(hits) else None
            if first is None: group = "never_alarmed"
            elif first < -12: group = "first_before_minus12"
            elif first <= -7: group = "first_minus12_to_minus7"
            elif first <= -1: group = "first_minus6_to_minus1"
            elif first <= 3: group = "first_onset_to_plus3"
            else: group = "first_after_plus3"
            pre6 = (relative >= -6) & (relative <= -1)
            useful = delta > 0
            useful_hits = np.flatnonzero(alarm & useful)
            pre6_hits = np.flatnonzero(alarm & pre6)
            longest = int(max(true_runs(alarm & pre6), default=0))
            record.update({"first_alarm_group": group, "first_alarm_relative_h": first,
                           "any_useful_alarm": int(bool(len(useful_hits))),
                           "pre6_alarm_hours": len(pre6_hits), "pre6_longest_alarm_run": longest,
                           "pre6_detected": int(bool(len(pre6_hits))),
                           "useful_first_lead_h": int(-relative[useful_hits[0]]) if len(useful_hits) else None,
                           "pre6_first_lead_h": int(-relative[pre6_hits[0]]) if len(pre6_hits) else None,
                           "early_first_then_useful": int(first is not None and first < -12 and len(useful_hits) > 0),
                           "early_first_then_pre6": int(first is not None and first < -12 and len(pre6_hits) > 0)})
            for r in range(-24, 4):
                mask = relative == r
                if mask.any():
                    curve_rows.append({"model": model, "split": split, "hospital": person.hospital,
                                       "relative_hour": r, "hours": int(mask.sum()), "alarms": int(alarm[mask].sum()),
                                       "loss_integer": int(regret[mask].sum()),
                                       "opportunity_integer": int(np.maximum(delta[mask], 0).sum())})
        else:
            record.update({"first_alarm_group": "nonseptic", "first_alarm_relative_h": None,
                           "any_useful_alarm": 0, "pre6_alarm_hours": 0, "pre6_longest_alarm_run": 0,
                           "pre6_detected": 0, "useful_first_lead_h": None, "pre6_first_lead_h": None,
                           "early_first_then_useful": 0, "early_first_then_pre6": 0})
            for length in true_runs(alarm):
                run_rows.append({"model": model, "split": split, "hospital": person.hospital,
                                 "patient_id": person.patient_id, "length": int(length)})
        # This is a descriptive score-distance audit, not a threshold sweep.
        ratio = probability / cutoff
        for phase in ("useful_early", "pre6", "onset", "nonseptic_false_alarm"):
            if phase == "nonseptic_false_alarm":
                mask = alarm if relative is None else np.zeros(len(y), bool)
                edges = [(1.,1.25,"1_to_1p25"),(1.25,2.,"1p25_to_2"),(2.,np.inf,"at_least_2")]
            else:
                mask = ((delta > 0) & ~alarm) if relative is not None else np.zeros(len(y), bool)
                if relative is not None:
                    if phase == "useful_early": mask &= relative < -6
                    elif phase == "pre6": mask &= (relative >= -6) & (relative < 0)
                    else: mask &= relative >= 0
                edges = [(0.,.5,"below_half"),(.5,.75,"half_to_0p75"),(.75,.9,"0p75_to_0p9"),(.9,1.,"0p9_to_threshold")]
            if not mask.any():
                continue
            for low, high, band in edges:
                included = mask & (ratio >= low) & (ratio < high)
                if not included.any():
                    continue
                score_rows.append({"model": model, "split": split, "hospital": person.hospital,
                                   "phase": phase, "ratio_band": band, "hours": int(included.sum()),
                                   "loss_integer": int(regret[included].sum())})
        person_rows.append(record)
    people = pd.DataFrame(person_rows)
    denominator = int(people.denominator_integer.sum())
    regret = int(people.regret_integer.sum()) / denominator
    utility = float(people.gain_integer.sum()/denominator)
    np.testing.assert_allclose(1-utility, regret, rtol=0, atol=1e-12)
    np.testing.assert_allclose(utility, expected_us, rtol=0, atol=1e-12)
    if checked:
        official_us = (official_observed-official_inactive)/(official_best-official_inactive)
        np.testing.assert_allclose(official_us, utility, rtol=0, atol=1e-11)
    return people, pd.DataFrame(run_rows), pd.DataFrame(curve_rows), pd.DataFrame(score_rows), checked


def aggregate(people, runs, curves, score_bands):
    summary, loss_rows, timing, pre6_rows, concentration, duration, score_rows, curve_rows, miss_coverage = [], [], [], [], [], [], [], [], []
    for hospital in ("ALL", "A", "B"):
        part = people if hospital == "ALL" else people.loc[people.hospital.eq(hospital)]
        model, split = part.model.iloc[0], part.split.iloc[0]
        prefix = {"model": model, "split": split, "hospital": hospital}
        denom = int(part.denominator_integer.sum())
        total_loss = int(part.regret_integer.sum())
        event = part.loc[part.has_event.eq(1)]
        nonsep = part.loc[part.has_event.eq(0)]
        fp = int(nonsep.alarm_hours.sum())
        summary.append({**prefix, "patients": len(part), "event_patients": len(event), "nonseptic_patients": len(nonsep),
                        "utility_score": float(part.gain_integer.sum()/denom), "regret": total_loss/denom,
                        "normalization_denominator": denom/180., "pre6_detected": int(event.pre6_detected.sum()),
                        "pre6_sensitivity": float(event.pre6_detected.mean()),
                        "missed_entire_useful_window": int((event.any_useful_alarm == 0).sum()),
                        "any_useful_alarm_patients": int(event.any_useful_alarm.sum()),
                        "never_alarmed_event_patients": int(event.first_alarm_group.eq("never_alarmed").sum()),
                        "first_too_early_patients": int(event.first_alarm_group.eq("first_before_minus12").sum()),
                        "early_first_then_useful": int(event.early_first_then_useful.sum()),
                        "early_first_then_pre6": int(event.early_first_then_pre6.sum()),
                        "nonseptic_alarm_patients": int(nonsep.alarm_hours.gt(0).sum()),
                        "nonseptic_patient_alarm_fraction": float(nonsep.alarm_hours.gt(0).mean()),
                        "nonseptic_false_alarm_hours": fp,
                        "nonseptic_hours": int(nonsep.hours.sum()),
                        "nonseptic_hour_alarm_fraction": fp/int(nonsep.hours.sum()),
                        "median_nonseptic_alarm_hours_if_alarmed": float(nonsep.loc[nonsep.alarm_hours.gt(0),"alarm_hours"].median()),
                        "median_pre6_lead_if_detected": float(event.pre6_first_lead_h.dropna().median()),
                        "median_first_any_alarm_relative_h": float(event.first_alarm_relative_h.dropna().median()),
                        "median_first_useful_alarm_lead_h": float(event.useful_first_lead_h.dropna().median())})
        for bucket in BUCKETS:
            loss = int(part[bucket+"_loss_integer"].sum())
            loss_rows.append({**prefix, "category": bucket, "category_zh": CHINESE[bucket],
                              "raw_regret": loss/180., "normalized_loss": loss/denom,
                              "share_of_total_regret": loss/total_loss,
                              "opportunity_hours": int(part[bucket+"_opportunity_hours"].sum()),
                              "loss_hours": int(part[bucket+"_loss_hours"].sum()),
                              "affected_patients": int(part[bucket+"_loss_integer"].gt(0).sum())})
        for group, frame in event.groupby("first_alarm_group", sort=True):
            timing.append({**prefix, "first_alarm_group": group, "patients": len(frame), "fraction": len(frame)/len(event),
                           "later_any_useful_alarm": int(frame.any_useful_alarm.sum()),
                           "later_pre6_alarm": int(frame.pre6_detected.sum())})
        for name, values in (("pre6_alarm_hours", event.pre6_alarm_hours), ("pre6_longest_alarm_run", event.pre6_longest_alarm_run)):
            for count in range(7):
                pre6_rows.append({**prefix, "measure": name, "count": count, "patients": int(values.eq(count).sum()),
                                  "fraction": float(values.eq(count).mean())})
        pre6_loss_total = int(event[BUCKETS[3]+"_loss_integer"].sum())
        for name, condition in (("no_pre6_alarm",event.pre6_alarm_hours.eq(0)),
                                ("partial_pre6_alarm",event.pre6_alarm_hours.between(1,5)),
                                ("all_six_pre6_alarm",event.pre6_alarm_hours.eq(6))):
            subgroup = event.loc[condition]
            subgroup_loss = int(subgroup[BUCKETS[3]+"_loss_integer"].sum())
            miss_coverage.append({**prefix, "coverage_group":name, "event_patients":len(subgroup),
                                  "event_fraction":len(subgroup)/len(event), "pre6_loss_integer":subgroup_loss,
                                  "pre6_normalized_loss":subgroup_loss/denom,
                                  "share_of_pre6_regret":subgroup_loss/pre6_loss_total,
                                  "never_alarmed_patients":int(subgroup.first_alarm_group.eq("never_alarmed").sum()),
                                  "first_alarmed_after_onset":int(subgroup.first_alarm_group.eq("first_onset_to_plus3").sum())})
        for population, subset in (("all_nonseptic", nonsep), ("alarmed_nonseptic", nonsep.loc[nonsep.alarm_hours.gt(0)])):
            ordered = subset.sort_values(["alarm_hours", "patient_id"], ascending=[False,True])
            for pct in (1,5,10):
                k = max(1,int(np.ceil(len(ordered)*pct/100)))
                concentration.append({**prefix, "population": population, "top_percent": pct, "patients": k,
                                      "false_alarm_hours": int(ordered.alarm_hours.iloc[:k].sum()),
                                      "fraction_of_all_nonseptic_false_alarm_hours": float(ordered.alarm_hours.iloc[:k].sum()/fp)})
        patient_bin = pd.cut(nonsep.hours, [0,24,48,72,120,np.inf], labels=["1_to_24","25_to_48","49_to_72","73_to_120","over_120"])
        for category in patient_bin.cat.categories:
            subset = nonsep.loc[patient_bin.eq(category)]
            duration.append({**prefix, "kind": "patient_record_length", "bin": category, "patients": len(subset),
                             "hours": int(subset.hours.sum()), "false_alarm_hours": int(subset.alarm_hours.sum()),
                             "false_alarm_hour_fraction": float(subset.alarm_hours.sum()/subset.hours.sum()) if len(subset) else None})
        rpart = runs if hospital == "ALL" else runs.loc[runs.hospital.eq(hospital)]
        for low, high, name in ((1,1,"1"),(2,2,"2"),(3,5,"3_to_5"),(6,11,"6_to_11"),(12,np.inf,"at_least_12")):
            subset = rpart.loc[rpart.length.ge(low) & rpart.length.le(high)]
            duration.append({**prefix, "kind": "continuous_false_alarm_episode", "bin": name,
                             "episodes": len(subset), "false_alarm_hours": int(subset.length.sum()),
                             "fraction_of_all_nonseptic_false_alarm_hours": float(subset.length.sum()/fp)})
        cpart = curves if hospital == "ALL" else curves.loc[curves.hospital.eq(hospital)]
        for relative, frame in cpart.groupby("relative_hour"):
            curve_rows.append({**prefix, "relative_hour": int(relative), "available_patients": int(frame.hours.sum()),
                               "alarmed_patients": int(frame.alarms.sum()), "alarm_fraction": float(frame.alarms.sum()/frame.hours.sum()),
                               "normalized_loss": float(frame.loss_integer.sum()/denom),
                               "normalized_opportunity": float(frame.opportunity_integer.sum()/denom)})
        spart = score_bands if hospital == "ALL" else score_bands.loc[score_bands.hospital.eq(hospital)]
        for (phase, band), frame in spart.groupby(["phase","ratio_band"]):
            phase_total = int(spart.loc[spart.phase.eq(phase),"loss_integer"].sum())
            score_rows.append({**prefix, "phase": phase, "ratio_band": band, "hours": int(frame.hours.sum()),
                               "normalized_loss": float(frame.loss_integer.sum()/denom),
                               "share_of_phase_regret": float(frame.loss_integer.sum()/phase_total) if phase_total else None})
    return {"overview":summary,"loss_components":loss_rows,"first_alarm_timing":timing,
            "pre6_alarm_coverage":pre6_rows,"false_alarm_concentration":concentration,
            "false_alarm_duration":duration,"score_distance":score_rows,"relative_time_curve":curve_rows,
            "miss_coverage":miss_coverage}


def measurement_audit(provenance):
    """TRAIN-only observability/time associations, with all thresholds frozen.

    Actual ICULOS at each prediction is available in real time. By contrast,
    actual future onset and final record length are retrospective audit groups,
    never suggested as online features or online threshold conditions.
    """
    patients, labels, scores, thresholds, _ = load_predictions(PRIMARY,"train",provenance)
    raw_path = available(BASE/"raw_float64.npy")
    raw = np.load(raw_path,mmap_mode="r")
    provenance[str(raw_path.relative_to(ROOT))] = sha(raw_path)
    bins = ("1_to_12","13_to_24","25_to_48","49_to_72","73_to_120","over_120")
    edges = [12,24,48,72,120]
    event_rows, current_rows = [], []
    for p in patients.itertuples(index=False):
        a,b=int(p.start),int(p.stop)
        values = raw[int(p.raw_start):int(p.raw_stop)]
        y, alarm = labels[a:b], scores[a:b]>=thresholds[a:b]
        delta, relative = delta_integer(y)
        current_bin = np.searchsorted(edges, values[:,39], side="left")
        event = relative is not None
        if event:
            onset = int(np.flatnonzero(y)[0])+6
            pre6 = (relative>=-6)&(relative<0)
            observed = np.isfinite(values[pre6,:34])
            hits = int(alarm[pre6].sum())
            coverage = "no_pre6_alarm" if not hits else "all_six_pre6_alarm" if hits==6 else "partial_pre6_alarm"
            event_rows.append({"hospital":p.hospital,"patient_id":p.patient_id,"coverage":coverage,
                               "pre6_detected":int(hits>0),"pre6_alarm_hours":hits,
                               "onset_iculos":float(values[onset,39]),
                               "onset_stratum":bins[np.searchsorted(edges,values[onset,39],side="left")],
                               "pre6_start_iculos":float(values[onset-6,39]),
                               "vital_observed_fraction":float(observed[:,:7].mean()),
                               "lab_observed_fraction":float(observed[:,8:].mean()),
                               "vital_variables_observed_in6h":int(observed[:,:7].any(axis=0).sum()),
                               "lab_variables_observed_in6h":int(observed[:,8:].any(axis=0).sum())})
        else:
            pre6 = np.zeros(len(y),bool)
        for index in np.unique(current_bin):
            mask = current_bin==index
            good = mask&pre6
            current_rows.append({"hospital":p.hospital,"patient_id":p.patient_id,"current_iculos_stratum":bins[index],
                                 "nonseptic_hours":int(mask.sum()) if not event else 0,
                                 "nonseptic_fp_hours":int(alarm[mask].sum()) if not event else 0,
                                 "nonseptic_patient_present":int(not event),
                                 "nonseptic_patient_alarmed":int(not event and alarm[mask].any()),
                                 "pre6_hours":int(good.sum()),"pre6_alarm_hours":int(alarm[good].sum()),
                                 "pre6_patient_present":int(good.any()),
                                 "pre6_patient_alarmed":int((good&alarm).any()),
                                 "pre6_opportunity_integer":int(np.maximum(delta[good],0).sum()),
                                 "pre6_regret_integer":int((np.maximum(delta[good],0)-delta[good]*alarm[good]).sum())})
    events, current = pd.DataFrame(event_rows),pd.DataFrame(current_rows)
    coverage_rows,onset_rows,current_summary = [],[],[]
    for hospital in ("ALL","A","B"):
        ev = events if hospital=="ALL" else events.loc[events.hospital.eq(hospital)]
        cu = current if hospital=="ALL" else current.loc[current.hospital.eq(hospital)]
        for coverage, frame in ev.groupby("coverage"):
            coverage_rows.append({"model":PRIMARY,"split":"train","hospital":hospital,"coverage":coverage,
                                  "patients":len(frame),"onset_iculos_median":float(frame.onset_iculos.median()),
                                  "onset_iculos_mean":float(frame.onset_iculos.mean()),
                                  "vital_observed_fraction_mean":float(frame.vital_observed_fraction.mean()),
                                  "vital_observed_fraction_median":float(frame.vital_observed_fraction.median()),
                                  "lab_observed_fraction_mean":float(frame.lab_observed_fraction.mean()),
                                  "lab_observed_fraction_median":float(frame.lab_observed_fraction.median()),
                                  "vital_variables_observed_in6h_mean":float(frame.vital_variables_observed_in6h.mean()),
                                  "lab_variables_observed_in6h_mean":float(frame.lab_variables_observed_in6h.mean())})
        for stratum in bins:
            frame = ev.loc[ev.onset_stratum.eq(stratum)]
            onset_rows.append({"model":PRIMARY,"split":"train","hospital":hospital,"onset_iculos_stratum":stratum,
                               "patients":len(frame),"pre6_detected":int(frame.pre6_detected.sum()),
                               "pre6_sensitivity":float(frame.pre6_detected.mean()) if len(frame) else None,
                               "pre6_all_six_alarm":int(frame.coverage.eq("all_six_pre6_alarm").sum()),
                               "vital_observed_fraction_mean":float(frame.vital_observed_fraction.mean()) if len(frame) else None,
                               "lab_observed_fraction_mean":float(frame.lab_observed_fraction.mean()) if len(frame) else None})
            frame = cu.loc[cu.current_iculos_stratum.eq(stratum)]
            total = {key:int(frame[key].sum()) for key in ("nonseptic_hours","nonseptic_fp_hours","nonseptic_patient_present",
                        "nonseptic_patient_alarmed","pre6_hours","pre6_alarm_hours","pre6_patient_present","pre6_patient_alarmed",
                        "pre6_opportunity_integer","pre6_regret_integer")}
            current_summary.append({"model":PRIMARY,"split":"train","hospital":hospital,"current_iculos_stratum":stratum,
                                    **total,"nonseptic_fp_hour_fraction":total["nonseptic_fp_hours"]/total["nonseptic_hours"] if total["nonseptic_hours"] else None,
                                    "nonseptic_patient_alarm_fraction":total["nonseptic_patient_alarmed"]/total["nonseptic_patient_present"] if total["nonseptic_patient_present"] else None,
                                    "pre6_hour_sensitivity":total["pre6_alarm_hours"]/total["pre6_hours"] if total["pre6_hours"] else None,
                                    "pre6_utility_capture_fraction":1-total["pre6_regret_integer"]/total["pre6_opportunity_integer"] if total["pre6_opportunity_integer"] else None})
    return {"train_measurement_by_coverage":coverage_rows,"train_onset_strata":onset_rows,
            "train_current_iculos_strata":current_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-validation", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    splits = ("train","validation","test") if args.include_validation else ("train","test")
    official = official_function()
    tables, provenance, checked = {}, {}, 0
    examples = []
    for model in MODELS:
        for split in splits:
            print(f"Auditing {model}/{split}", flush=True)
            people,runs,curves,bands,n_checked = audit(model,split,official,provenance)
            checked += n_checked
            for name, rows in aggregate(people,runs,curves,bands).items():
                tables.setdefault(name,[]).extend(rows)
            if model == PRIMARY:
                for condition, name, metric in ((people.has_event.eq(0),"top_nonseptic_FP","alarm_hours"),
                                                (people.has_event.eq(1),"top_septic_early_FP",BUCKETS[1]+"_loss_integer"),
                                                (people.has_event.eq(1),"top_useful_window_misses","useful_missed_loss_integer")):
                    chosen = people.loc[condition].sort_values([metric,"patient_id"],ascending=[False,True]).head(20).copy()
                    chosen["example_selection"] = name
                    examples.extend(chosen.to_dict("records"))
    tables.update(measurement_audit(provenance))
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(REPORT / f"{name}.csv", index=False)
    pd.DataFrame(examples).to_csv(REPORT / "patient_examples.csv", index=False)
    write_json(REPORT / "audit_manifest.json", {"primary_model":PRIMARY,"comparison_models":list(MODELS),
        "splits":list(splits),"source_sha256":sha(__file__),"input_sha256":provenance,
        "official_primary_patient_checks":checked,"formula":"regret = [max(delta,0)-delta*alarm] / sum(max(delta,0)) = 1-US",
        "training_thresholds":"Per outer fold inner-calibration thresholds, unchanged",
        "test_thresholds":"Frozen validation thresholds, unchanged",
        "no_ensemble":True,"new_fits":0,"new_threshold_search":False,"test_used_for_policy_selection":False,
        "record_length_diagnostic_only":True,
        "observability_audit":"TRAIN only; onset ICU hour and record length are retrospective groups, not input features. Current ICULOS strata use only elapsed time available at each prediction.",
        "interpretation":"Training OOF supports next experiments; fixed test is historically exposed exploratory diagnosis only.",
        "run_definition":"Consecutive positive hourly decisions; no cooldown, no postprocessing applied."})
    write_report(tables)
    print("Completed official utility regret audit",flush=True)


def write_report(tables):
    components = pd.DataFrame(tables["loss_components"])
    overview = pd.DataFrame(tables["overview"])
    rows = components.loc[components.model.eq(PRIMARY)&components.hospital.eq("ALL")]
    text=["# 单模型 US 损失审计", "", "主参考模型预先指定为789维、63叶LightGBM。比较563维基线、791维LightGBM及优化XGBoost，不包含集成。",
          "训练采用原三折OOF分数及各折内部校准阈值；测试采用已冻结验证阈值。此处不重训、不重选阈值。下一轮方向依据训练OOF，测试仅作固定诊断。", "",
          "## 效用损失的定义", "", "损失为相对理想逐小时决策的regret：`[max(ΔU,0)−ΔU×报警]/Σmax(ΔU,0)`，合计恰为`1−US`。它不是普通分类错误率。原标签0的−11至−7小时，未报警也会损失正效用。", "",
          "| 损失来源 | 训练OOF损失点 | 占训练总损失 | 测试损失点 | 占测试总损失 |", "|---|---:|---:|---:|---:|"]
    for category in BUCKETS:
        train=rows.loc[rows.split.eq("train")&rows.category.eq(category)].iloc[0]
        test=rows.loc[rows.split.eq("test")&rows.category.eq(category)].iloc[0]
        text.append(f"| {CHINESE[category]} | {train.normalized_loss:.6f} | {train.share_of_total_regret:.2%} | {test.normalized_loss:.6f} | {test.share_of_total_regret:.2%} |")
    text += ["", "误报包括非脓毒症患者报警，以及最终发病患者在发病前超过12小时的报警；后三项是正效用窗口没有报警的机会损失。发病后0至3小时的漏报是官方评分的重要损失，但不能把补回这些分数解释为更早预警。", "",
             "## 命中与首报警时机", "", "| 集合 | US | 发病前6小时至少1次命中 | 整个正效用窗口未命中 | 首报警早于−12小时 | 其中后来6小时窗口命中 |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("train","test"):
        r=overview.loc[overview.model.eq(PRIMARY)&overview.hospital.eq("ALL")&overview.split.eq(split)].iloc[0]
        text.append(f"| {split} | {r.utility_score:.6f} | {int(r.pre6_detected)}/{int(r.event_patients)} ({r.pre6_sensitivity:.1%}) | {int(r.missed_entire_useful_window)} | {int(r.first_too_early_patients)} | {int(r.early_first_then_pre6)} |")
    text += ["", "首报警很早并不代表有效窗口命中；首报警过早后持续命中，与只有一次无效早报，应分开解释。完整分布见first_alarm_timing.csv及pre6_alarm_coverage.csv；相对发病逐小时曲线见relative_time_curve.csv。", "",
             "误报持续长度以连续逐小时阳性段定义，不加入报警冷却。集中度、记录长度及分数距阈值的分布分别见false_alarm_concentration.csv、false_alarm_duration.csv和score_distance.csv。", "",
             "记录总长度是事后诊断分组，不能作为模型输入或在线阈值条件；只有当前已经过的ICU时间可在预测时获得。全部分解按医院A/B另存，归一化分母为对应医院自己的理想增益；不要把A/B的归一化损失直接相加。patient_examples.csv是按各类损失最高确定性选出的复核案例，不是代表性抽样。", "",
             "训练队列的发病记录没有发病后+3小时以后的小时；不能用大量晚期零效用阳性解释本队列结果。官方评分器逐患者核验主模型的训练与测试regret及分母；其余单模型与冻结US对照。", "",
             "这些结果依赖现有完整记录标签，且测试患者此前已被查看；属于探索性诊断。没有用测试集筛选新的门控/平滑策略。", ""]
    primary_filter = lambda frame: frame.loc[frame.model.eq(PRIMARY)&frame.split.eq("train")&frame.hospital.eq("ALL")]
    cover = primary_filter(pd.DataFrame(tables["miss_coverage"]))
    bands = primary_filter(pd.DataFrame(tables["score_distance"]))
    concentration = primary_filter(pd.DataFrame(tables["false_alarm_concentration"]))
    duration = primary_filter(pd.DataFrame(tables["false_alarm_duration"]))
    no_hit = cover.loc[cover.coverage_group.eq("no_pre6_alarm")].iloc[0]
    low = bands.loc[bands.phase.eq("pre6")&bands.ratio_band.eq("below_half")].iloc[0]
    near = bands.loc[bands.phase.eq("pre6")&bands.ratio_band.eq("0p9_to_threshold")].iloc[0]
    top5 = concentration.loc[concentration.population.eq("all_nonseptic")&concentration.top_percent.eq(5)].iloc[0]
    isolated = duration.loc[duration.kind.eq("continuous_false_alarm_episode")&duration.bin.eq("1")].iloc[0]
    long = duration.loc[duration.kind.eq("continuous_false_alarm_episode")&duration.bin.eq("at_least_12")].iloc[0]
    measurement = primary_filter(pd.DataFrame(tables["train_measurement_by_coverage"]))
    never = measurement.loc[measurement.coverage.eq("no_pre6_alarm")].iloc[0]
    all_six = measurement.loc[measurement.coverage.eq("all_six_pre6_alarm")].iloc[0]
    text += ["## 依据训练OOF的下一轮优先级", "",
             f"1. **先改善整段低风险分数的漏检。**{int(no_hit.event_patients)}例在六小时窗口一次也没报警，贡献{no_hit.share_of_pre6_regret:.1%}的该窗口漏报损失；"
             f"其中{int(no_hit.never_alarmed_patients)}例全程没有报警。漏报效用中{low.share_of_phase_regret:.1%}的分数低于各折阈值一半，只有{near.share_of_phase_regret:.1%}位于阈值的90%至100%。"
             "因此小范围挪阈值或仅修补短暂断续难以解决主体。优先尝试相对患者早期基线的偏离、病情新近恶化与稳定异常的区分；所有基线必须只使用已发生历史。", "",
             f"2. **针对持续误报，而非只消除尖峰。**非脓毒症患者的误报小时中，孤立一小时段占{isolated.fraction_of_all_nonseptic_false_alarm_hours:.1%}，"
             f"长度≥12小时段占{long.fraction_of_all_nonseptic_false_alarm_hours:.1%}；按误报小时排序的前5%非脓毒症患者贡献{top5.fraction_of_all_nonseptic_false_alarm_hours:.1%}。"
             "可优先做明确ICULOS/HospAdmTime及其派生特征的消融，检查长期高风险与当前发病风险混淆；该诊断本身不能证明时钟特征导致误报。", "",
             f"训练中，pre6窗口无报警组与全6小时报警组的生命体征实际观测率为{never.vital_observed_fraction_mean:.1%}与{all_six.vital_observed_fraction_mean:.1%}，"
             f"化验观测率为{never.lab_observed_fraction_mean:.1%}与{all_six.lab_observed_fraction_mean:.1%}；其推算发病ICULOS中位数却为{never.onset_iculos_median:.0f}与{all_six.onset_iculos_median:.0f}小时。"
             "这些是观测关联，不能把漏报简单归于窗口内普遍缺失，也不能仅凭其断言时钟特征为因果原因。"
             "应结合当前已知ICULOS分层的误报/窗口命中（train_current_iculos_strata.csv）、发病时刻回顾分组（train_onset_strata.csv）及显式时钟特征消融加以验证。", "",
             "3. **低成本的时间策略只作训练内受控实验。**短期分数保留可检验漏检断续，连续命中门控可检验误报削减，但会延迟报警或增加错误持续。"
             "每折策略及阈值应只用该折内部校准患者选择，再在外折评价；不能在现有测试结果上挑规则。新的效用目标实验应固定早停标准，避免同时改目标和早停所产生的解释混淆。", ""]
    (REPORT/"report.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
