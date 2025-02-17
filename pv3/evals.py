
def precision_value(tp: int, fp: int) -> float:
    if tp + fp == 0: return 0.0
    return float(tp) / float(tp + fp)

def recall_value(tp: int, fn: int) -> float:
    if tp + fn == 0: return 0.0
    return float(tp) / float(tp + fn)

def count_tp(pred: list, actual: list):
    return len(set(pred).intersection(set(actual)))

def count_fp(pred: list, actual: list):
    return len(set(actual).difference(set(pred)))

def count_fn(pred: list, actual: list):
    return len(set(pred).difference(set(actual)))

def precision(pred: list, actual: list) -> float:
    tp = count_tp(pred, actual)
    fp = count_fp(pred, actual)
    return precision_value(tp, fp)

def recall(pred: list, actual: list) -> float:
    tp = count_tp(pred, actual)
    fn = count_fn(pred, actual)
    return recall_value(tp, fn)

def fbeta(beta: float, precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0: return 0.0
    return (1 + beta*beta) * ((precision * recall) / (beta*beta * precision + recall))
