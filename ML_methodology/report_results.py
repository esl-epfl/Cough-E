import glob
import os
import pandas as pd
import numpy as np


## Helper functions
# Get duration-based metrics
def get_metrics_per_threshold(df2):
    ses = []
    ppvs = []
    f1s = []
    thresholds = np.unique(df2.threshold.to_numpy())
    for th in thresholds:
        true_sum = np.sum(df2.loc[(df2.threshold == th)].true_dur)
        pred_sum = np.sum(df2.loc[(df2.threshold == th)].pred_dur)
        tp_sum = np.sum(df2.loc[(df2.threshold == th)].tp_dur)
        se_overall = tp_sum / true_sum if true_sum > 0 else 0
        ses.append(se_overall)
        ppv_overall = tp_sum / pred_sum if pred_sum > 0 else 0
        ppvs.append(ppv_overall)
        f1_overall = (
            2 * se_overall * ppv_overall / (se_overall + ppv_overall)
            if (se_overall + ppv_overall) > 0
            else 0
        )
        f1s.append(f1_overall)
    return (
        np.array(ses).reshape(-1, 1),
        np.array(ppvs).reshape(-1, 1),
        np.array(f1s).reshape(-1, 1),
    )


def get_evb_metrics_per_threshold(df2):
    ses = []
    ppvs = []
    f1s = []
    thresholds = np.unique(df2.threshold.to_numpy())
    for th in thresholds:
        TP = np.sum(df2.loc[(df2.threshold == th)].tp_evt)
        FP = np.sum(df2.loc[(df2.threshold == th)].fp_evt)
        FN = np.sum(df2.loc[(df2.threshold == th)].fn_evt)
        se_evt = TP / (TP + FN) if (TP + FN) > 0 else 0
        ppv_evt = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_evt = (
            2 * se_evt * ppv_evt / (se_evt + ppv_evt) if (se_evt + ppv_evt) > 0 else 0
        )
        ses.append(se_evt)
        ppvs.append(ppv_evt)
        f1s.append(f1_evt)
    subj = np.unique(df2.subject)
    return (
        np.array(ses).reshape(-1, 1),
        np.array(ppvs).reshape(-1, 1),
        np.array(f1s).reshape(-1, 1),
    )


if __name__ == "__main__":
    # Choose folder in outputs where the CV results are saved
    edge_ai_results_folder = "outputs/2024-03-08/11-28-12/" 

    df_folds = list()
    for name in os.listdir(edge_ai_results_folder):
        if name.split("_")[0] == "fold":
            df_fold = pd.read_csv(edge_ai_results_folder + name + "/results.csv")
            df_folds.append(df_fold)

    thresholds = np.arange(0, 1.01, 0.01)

    ses_s = list()
    ppvs_s = list()
    f1s_s = list()
    for i, fold in enumerate(df_folds):
        ses_0, ppvs_0, f1s_0 = get_evb_metrics_per_threshold(fold)
        ses_s.append(ses_0)
        ppvs_s.append(ppvs_0)
        f1s_s.append(f1s_0)

    all_f1s = np.concatenate(f1s_s, axis=1)
    all_ses = np.concatenate(ses_s, axis=1)
    all_ppvs = np.concatenate(ppvs_s, axis=1)

    print("***EVENT-BASED METRICS***")
    print(
        "Best threshold: {0}, Average F-1 score: {1} +/- {2}".format(
            thresholds[np.argmax(all_f1s.mean(1))],
            np.max(all_f1s.mean(1)),
            all_f1s.std(1)[np.argmax(all_f1s.mean(1))],
        )
    )
    print(
        "SE: {0}, PPV: {1}".format(
            all_ses.mean(1)[np.argmax(all_f1s.mean(1))],
            all_ppvs.mean(1)[np.argmax(all_f1s.mean(1))],
        )
    )
