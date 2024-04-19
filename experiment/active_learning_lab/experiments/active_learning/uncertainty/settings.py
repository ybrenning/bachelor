from collections import OrderedDict


DATASETS = OrderedDict({
    'ag-news': 'AGN',
    'dbp': 'DBP',
    'imdb': 'IMDB',
    'sst-2': 'SST-2',
    'subj': 'SUBJ',
    'trec': 'TREC-6',
    'yah': 'YAH'})


CONFIG_IDS = {'kimcnn': 'KimCNN', 'distilroberta-base': 'DistilRoBERTa',
              'bert-base-uncased': 'BERT'}


QUERY_STRATEGIES = OrderedDict({'ulb-kl': 'UL', 'lc-ent': 'PE', 'lc-bt': 'BT', 'badge': 'BA',
                                'gc': 'GC', 'cal': 'CA', 'random': 'RS'})

# was not recorded
QUERY_SIZES = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]