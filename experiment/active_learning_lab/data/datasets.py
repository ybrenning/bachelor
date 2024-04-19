from enum import Enum


class UnknownDataSetException(ValueError):
    pass


class UnknownDataSetTypeException(ValueError):
    pass


class DatasetReaderNotFoundException(ValueError):
    pass


class DataSets(Enum):
    TWENTY_NEWSGROUPS = '20-newsgroups'
    AG_NEWS = 'ag-news'
    AMAZON_REVIEW_POLARITY = 'amazon-review-polarity'
    YELP_REVIEW_POLARITY = 'yelp-review-polarity'
    RCV1_V2 = 'rcv1-v2'
    TREC = 'trec'
    SST = 'sst'
    MR = 'mr'
    SUBJ = 'subj'
    CR = 'CR'
    YAH = 'yah'  # yahoo-answers
    DBP = 'dbp'  # dbpedia
    OOS_EVAL = 'oos-eval'
    SST2 = 'sst-2'
    GO_EMOTIONS = 'go-emotions'
    IMDB = 'imdb'

    @staticmethod
    def from_str(enum_str: str):
        if enum_str == '20-newsgroups':
            return DataSets.TWENTY_NEWSGROUPS
        elif enum_str == 'ag-news':
            return DataSets.AG_NEWS
        elif enum_str == 'amazon-review-polarity':
            return DataSets.AMAZON_REVIEW_POLARITY
        elif enum_str == 'yelp-review-polarity':
            return DataSets.YELP_REVIEW_POLARITY
        elif enum_str == 'rcv1-v2':
            return DataSets.RCV1_V2
        elif enum_str == 'trec':
            return DataSets.TREC
        elif enum_str == 'sst':
            return DataSets.SST
        elif enum_str == 'mr':
            return DataSets.MR
        elif enum_str == 'subj':
            return DataSets.SUBJ
        elif enum_str == 'cr':
            return DataSets.CR
        elif enum_str == 'yah':
            return DataSets.YAH
        elif enum_str == 'dbp':
            return DataSets.DBP
        elif enum_str == 'oos-eval':
            return DataSets.OOS_EVAL
        elif enum_str == 'sst-2':
            return DataSets.SST2
        elif enum_str == 'go-emotions':
            return DataSets.GO_EMOTIONS
        elif enum_str == 'imdb':
            return DataSets.IMDB

        raise UnknownDataSetException('Enum DataSets does not contain the given element: '
                                      '\'{}\''.format(enum_str))


class DataSetType(Enum):
    TENSOR_PADDED_SEQ = 'tps'
    BOW = 'bow'
    RAW = 'raw'
    HUGGINGFACE = 'huggingface'
    SETFIT = 'setfit'

    @staticmethod
    def from_str(enum_str: str):
        if enum_str == 'tps':
            return DataSetType.TENSOR_PADDED_SEQ
        elif enum_str == 'bow':
            return DataSetType.BOW
        elif enum_str == 'raw':
            return DataSetType.RAW
        elif enum_str == 'huggingface':
            return DataSetType.HUGGINGFACE
        elif enum_str == 'setfit':
            return DataSetType.SETFIT

        raise UnknownDataSetTypeException(
            'Enum DataSetType does not contain the given element: ''\'{}\''.format(enum_str))
