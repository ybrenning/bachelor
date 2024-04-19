from collections import OrderedDict


def get_stopping_criteria_from_str(sc_strings, num_classes) -> OrderedDict:
    criteria = OrderedDict()
    for sc_string in sc_strings:
        criteria[sc_string] = _get_stopping_criterion_from_str(sc_string, num_classes)

    if len(criteria) == 0:
        return None

    return criteria


def _get_stopping_criterion_from_str(sc_string, num_classes):
    if sc_string == 'kappa':
        from small_text.stopping_criteria.kappa import KappaAverage
        return KappaAverage(num_classes)
    elif sc_string == 'kappa_k99':
        from small_text.stopping_criteria.kappa import KappaAverage
        return KappaAverage(num_classes, kappa=0.99)
    elif sc_string == 'kappa_k95':
        from small_text.stopping_criteria.kappa import KappaAverage
        return KappaAverage(num_classes, kappa=0.95)
    elif sc_string == 'kappa_k90':
        from small_text.stopping_criteria.kappa import KappaAverage
        return KappaAverage(num_classes, kappa=0.9)
    elif sc_string == 'delta_f':
        from small_text.stopping_criteria.base import DeltaFScore
        return DeltaFScore(num_classes)
    elif sc_string == 'ouncertainty_005':
        from small_text.stopping_criteria import OverallUncertainty
        return OverallUncertainty(num_classes, threshold=0.05)
    elif sc_string == 'ouncertainty_010':
        from small_text.stopping_criteria import OverallUncertainty
        return OverallUncertainty(num_classes, threshold=0.1)
    elif sc_string == 'ouncertainty_015':
        from small_text.stopping_criteria import OverallUncertainty
        return OverallUncertainty(num_classes, threshold=0.15)
    elif sc_string == 'cchange_005':
        from small_text.stopping_criteria import ClassificationChange
        return ClassificationChange(num_classes, threshold=0.05)
    elif sc_string == 'cchange_010':
        from small_text.stopping_criteria import ClassificationChange
        return ClassificationChange(num_classes, threshold=0.1)
    elif sc_string == 'cchange_015':
        from small_text.stopping_criteria import ClassificationChange
        return ClassificationChange(num_classes, threshold=0.15)
    else:
        raise ValueError('Unknown stopping criterion string: ' + str)
