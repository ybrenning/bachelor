import mlflow
import numpy as np
import pandas as pd

from active_learning_lab.utils.results import (
        assemble_df,
        check_for_reproduciblity,
        check_for_duplicates,
        get_results
)


DATASETS = {
    "mr": "MR",
    "ag-news": "AGN",
    "trec": "TREC"
}

CLASSIFIERS = {
    "transformer": "BERT",
    "setfit": "SetFit",
}

QUERY_STRATEGIES = {
    "random": "RS",
    "lc-bt": "BT",
    "gc": "CS",
    "gc-tsne": "CS-TSNE",
    "wgc": "WCS",
    "rwgc": "RCS"
}


def create_table(dfs, table_type):
    table_content = {}

    for q in QUERY_STRATEGIES.keys():
        table_content[q] = {}
        for c in CLASSIFIERS.keys():
            table_content[q][c] = {}

    for df in dfs:
        dataset = df["dataset_name"][0]
        classifier = df["classifier"][0]
        query_strategy = df["query_strategy"][0]

        final_vals = df.iloc[20::21] if table_type == 'acc' else df

        if table_type == 'acc' and query_strategy == 'gc-pca-tsne':
            print(dataset, classifier, query_strategy)
            print(round(final_vals["test_acc"].mean(), 3), "+-", round(final_vals["test_acc"].std(), 3))
            continue
        elif table_type == 'auc' and query_strategy == 'gc-pca-tsne':
            print(dataset, classifier, query_strategy)
            print(round(final_vals["auc_test_acc"].mean(), 3), "+-", round(final_vals["auc_test_acc"].std(), 3))
            continue

        if table_type == 'acc':
            table_content[query_strategy][classifier][dataset] = (
                round(final_vals["test_acc"].mean(), 3), round(final_vals["test_acc"].std(), 3)
            )
        elif table_type == 'auc':
            table_content[query_strategy][classifier][dataset] = (
                round(final_vals["auc_test_acc"].mean(), 3), round(final_vals["auc_test_acc"].std(), 3)
            )
        else:
            raise ValueError('Invalid table type')

    for d in DATASETS.keys():
        print(
                    "\multirow{2}{*}{" + DATASETS[d] + \
                        "}  & BERT & "+ str(table_content.get("random").get("transformer").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("random").get("transformer").get(d, (0, 0))[1])  + \
                        " & " + str(table_content.get("lc-bt").get("transformer").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("lc-bt").get("transformer").get(d, (0, 0))[1]) + \
                        " & \\bfseries " + str(table_content.get("gc").get("transformer").get(d, (0, 0))[0]) + \
                        " & \\bfseries " + str(table_content.get("gc").get("transformer").get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("gc-tsne", {}).get("transformer", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("gc-tsne", {}).get("transformer", {}).get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("wgc", {}).get("transformer", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("wgc", {}).get("transformer", {}).get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("rwgc", {}).get("transformer", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("rwgc", {}).get("transformer", {}).get(d, (0, 0))[1]) + \
                    "\\\\ \n & SetFit & " + str(table_content.get("random").get("setfit").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("random").get("setfit").get(d, (0, 0))[1]) + \
                        " & \\bfseries " + str(table_content.get("lc-bt").get("setfit").get(d, (0, 0))[0]) + \
                        " & \\bfseries " + str(table_content.get("lc-bt").get("setfit").get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("gc").get("setfit").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("gc").get("setfit").get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("gc-tsne", {}).get("setfit", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("gc-tsne", {}).get("setfit", {}).get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("wgc", {}).get("setfit", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("wgc", {}).get("setfit", {}).get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("rwgc", {}).get("setfit", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("rwgc", {}).get("setfit", {}).get(d, (0, 0))[1]) + \
                    " \\\\"
        )


def check_nth_result(df, n, type="acc"):
    auc = "" if type == "acc" else "auc_"
    final_vals = df[n] if type == 'auc' else df[n].iloc[20::21]

    if type == "acc":
        print(df[n].iloc[0]["dataset_name"], df[n].iloc[0]["classifier"], df[n].iloc[0]["query_strategy"])
    else:
        print(df[n]["dataset_name"].iloc[0], df[n]["classifier"].iloc[0], df[n]["query_strategy"].iloc[0])

    print(type, final_vals[f"{auc}test_acc"].mean())
    print(type, final_vals[f"{auc}test_acc"].std())


def main():
    results = get_results('yb-coresets')
    # results = check_for_duplicates(results)
    results = check_for_reproduciblity(results)

    df_acc = assemble_df(results, 'results.csv')
    df_auc = assemble_df(results, 'auc.csv')
    check_nth_result(df_acc, 1)
    check_nth_result(df_auc, 1, type="auc")

    # create_table(df_acc, table_type='acc')
    # create_table(df_auc, table_type='auc')


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///mnt/ceph/storage/data-tmp/current/yb63tadu/mlruns")
    main()
