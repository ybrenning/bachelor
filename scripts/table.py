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
        "mr": "Movie Review",
        "ag-news": "AG's News",
        "trec": "TREC-6"
}

CLASSIFIERS = {
        "transformer": "BERT",
        "setfit": "SetFit",
}

QUERY_STRATEGIES = {
        "random": "RS",
        "lc-bt": "BT",
        "gc": "CS",
        "MY_STRATEGY": "My Strategy"
}

def main():
    results = get_results('yb-coresets')
    results = check_for_duplicates(results)
    results = check_for_reproduciblity(results)

    df_acc = assemble_df(results, 'results.csv')
    # df_auc = assemble_df(results, 'auc.csv')

    table_content = {}

    for q in QUERY_STRATEGIES.keys():
        table_content[q] = {}
        for c in CLASSIFIERS.keys():
            table_content[q][c] = {}

    for df in df_acc:
        dataset = df["dataset_name"][0]
        classifier = df["classifier"][0]
        query_strategy = df["query_strategy"][0]

        table_content[query_strategy][classifier][dataset] = (round(df["test_acc"].mean(), 3), round(df["test_acc"].std(), 3))

        # print(f"dataset: {dataset}, classifier: {classifier}, query_strategy: {query_strategy}")
        # print("mean", round(df["test_acc"].mean(), 3))
        # print("std", round(df["test_acc"].std(), 3))
        # print()


    for d in DATASETS.keys():
        print(
                    "\multirow{2}{*}{" + DATASETS[d] + \
                        "}  & BERT & "+ str(table_content.get("random").get("transformer").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("random").get("transformer").get(d, (0, 0))[1])  + \
                        " & " + str(table_content.get("lc-bt").get("transformer").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("lc-bt").get("transformer").get(d, (0, 0))[1]) + \
                        " & \\bfseries " + str(table_content.get("gc").get("transformer").get(d, (0, 0))[0]) + \
                        " & \\bfseries " + str(table_content.get("gc").get("transformer").get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("MY_STRATEGY", {}).get("transformer", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("MY_STRATEGY", {}).get("transformer", {}).get(d, (0, 0))[1]) + \
                        "\\\\ \n & SetFit & " + str(table_content.get("random").get("setfit").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("random").get("setfit").get(d, (0, 0))[1]) + \
                        " & \\bfseries " + str(table_content.get("lc-bt").get("setfit").get(d, (0, 0))[0]) + \
                        " & \\bfseries " + str(table_content.get("lc-bt").get("setfit").get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("gc").get("setfit").get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("gc").get("setfit").get(d, (0, 0))[1]) + \
                        " & " + str(table_content.get("MY_STRATEGY", {}).get("setfit", {}).get(d, (0, 0))[0]) + \
                        " & " + str(table_content.get("MY_STRATEGY", {}).get("setfit", {}).get(d, (0, 0))[1]) + " \\\\"
        )


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///mnt/ceph/storage/data-tmp/current/yb63tadu/mlruns")
    main()
