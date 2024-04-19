import mlflow
import numpy as np
import pandas as pd

from active_learning_lab.utils.results import (
    assemble_df,
    get_results,
    check_for_duplicates,
    check_for_reproduciblity
)

CLASSIFIER_KEY_TO_DISPLAY_NAME = {
    "transformer": "BERT",
    "setfit": "SetFit"
}

DATASET_KEY_TO_DISPLAY_NAME = {
    "mr": "Movie Review",
    "ag-news": "AG's News"
}

QUERY_STRATEGY_KEY_TO_DISPLAY_NAME = {
    "random": "Random"
}


def main():
    results = get_results('yb-coresets')
    results = check_for_duplicates(results)
    results = check_for_reproduciblity(results)

    df_acc = assemble_df(results, 'results.csv')

    df_auc = assemble_df(results, 'auc.csv')
    ranking_summary(df_acc, df_auc)


def ranking_summary(df_acc, df_auc):

    df_acc_unaltered = df_acc
    df_acc_unaltered = df_acc_unaltered.set_index(['dataset_name', 'classifier', 'query_strategy'])

    df_acc = df_acc[df_acc['query_id'] == 20]

    df_acc = df_acc[df_acc['query_strategy'].isin(list(QUERY_STRATEGY_KEY_TO_DISPLAY_NAME.keys()))]
    df_acc = df_acc.set_index(['dataset_name', 'classifier', 'query_strategy'])

    df_auc = df_auc[df_auc['query_strategy'].isin(list(QUERY_STRATEGY_KEY_TO_DISPLAY_NAME.keys()))]
    df_auc = df_auc.set_index(['dataset_name', 'classifier', 'query_strategy'])

    num_runs = df_acc['run_id'].unique().shape[0]
    num_entries = len(df_auc) // num_runs
    df_auc['run_id'] = np.array(list(range(1, num_runs + 1)) * num_entries)

    idx = pd.IndexSlice

    for i, c in enumerate(['transformer', 'setfit']):

        for j, (q, q_label) in enumerate(QUERY_STRATEGY_KEY_TO_DISPLAY_NAME.items()):

            ranks = []
            ranks_auc = []

            for d in DATASET_KEY_TO_DISPLAY_NAME.keys():

                q_index = get_q_index(df_acc, d, c, q)

                #print(f'{d} {CLASSIFIER_KEY_TO_DISPLAY_NAME[c]} {q_label} ({q_index})')
                #print(df_acc.loc[idx[d, c, :], :].reset_index()['query_strategy'].unique())
                gb = df_acc.loc[idx[d, c, :], :].groupby('run_id', sort=False)
                values = np.array([sub_df['test_acc'] for _, sub_df in gb])
                values = np.mean(values, axis=0)
                mean_acc = df_acc.loc[idx[:, c, q], :]['test_acc'].mean()

                # query_time_sec, update_time_sec
                mean_qtime = df_acc_unaltered.loc[idx[:, c, q], :]['query_time_sec'].sum()
                mean_qtime_m, mean_qtime_s = divmod(int(mean_qtime), 60)
                mean_qtime_h, mean_qtime_m = divmod(mean_qtime_m, 60)

                mean_ttime = df_acc_unaltered.loc[idx[:, c, q], :]['update_time_sec'].sum()
                mean_ttime_m, mean_ttime_s = divmod(int(mean_ttime), 60)
                mean_ttime_h, mean_ttime_m = divmod(mean_ttime_m, 60)

                rank_matrix = np.apply_along_axis(lambda x: get_rank(x, np.argsort(x)), 0, values)
                ranks.append(rank_matrix[q_index])

                q_index = get_q_index(df_auc, d, c, q)
                gb_auc = df_auc.loc[idx[d, c, :], :].groupby('run_id', sort=False)
                values_auc = np.array([sub_df['auc_test_acc'] for _, sub_df in gb_auc])
                values_auc = np.mean(values_auc, axis=0)
                mean_auc = df_auc.loc[idx[:, c, q], :]['auc_test_acc'].mean()

                rank_matrix_auc = np.apply_along_axis(lambda x: get_rank(x, np.argsort(x)), 0, values_auc)
                ranks_auc.append(rank_matrix_auc[q_index])

            avg_rank_acc = np.mean(ranks)
            avg_rank_auc = np.mean(ranks_auc)

            if j == 0:
                print('\multirow{8}{*}{\\rotatebox{90}{'+ CLASSIFIER_KEY_TO_DISPLAY_NAME[c]  + '}}' + f' & {q_label}  & {avg_rank_acc:.2f} & {avg_rank_auc:.2f} & {mean_acc:.3f} & {mean_auc:.3f}\\\\')
            else:
                print(f' & {q_label}  & {avg_rank_acc:.2f} & {avg_rank_auc:.2f} & {mean_acc:.3f} & {mean_auc:.3f}\\\\')

        print('\\midrule\n')


def get_rank(x, argsorted):
    ranks = np.array([
        argsorted.shape[0] - np.where(argsorted == i)[0][0]
        for i in range(argsorted.shape[0])
    ])
    ranks_orig = np.copy(ranks)
    # this is inefficient but I need a quick fix
    for i in range(argsorted.shape[0]):
        # i_argsorted = np.where(argsorted == 0)[0]
        idx_where = np.where(x == x[argsorted[i]])[0]
        if ranks[i] == ranks_orig[i] and idx_where.shape[0] > 1:
            ranks[idx_where] = ranks[idx_where].min()

    return ranks


def get_q_index(df, d, c, q):
    idx = pd.IndexSlice

    # I don't understand this but the order can vary
    query_strategies_local = df.loc[idx[d, c, :], :].reset_index()['query_strategy'].unique()

    q_where = np.where(query_strategies_local == q)
    assert len(q_where) == 1
    q_index = q_where[0][0]

    return q_index


if __name__ == '__main__':
    main()
