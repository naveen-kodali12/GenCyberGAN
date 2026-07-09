import numpy as np


def assign_window_label(labels, rule='any_attack', benign_label=0):
    labels = np.asarray(labels)
    if rule == 'majority':
        vals, counts = np.unique(labels, return_counts=True)
        return vals[np.argmax(counts)]
    if rule == 'last':
        return labels[-1]
    non_benign = labels[labels != benign_label]
    if len(non_benign) > 0:
        vals, counts = np.unique(non_benign, return_counts=True)
        return vals[np.argmax(counts)]
    return benign_label


def create_temporal_windows(X, y, T=30, stride=5, label_rule='any_attack', benign_label=0):
    X = np.asarray(X, dtype='float32')
    y = np.asarray(y)
    windows, labels, starts = [], [], []
    if len(X) < T:
        pad = np.repeat(X[-1:], T-len(X), axis=0) if len(X) else np.zeros((T, X.shape[1]), dtype='float32')
        Xp = np.vstack([X, pad]) if len(X) else pad
        yp = np.concatenate([y, np.repeat(y[-1], T-len(y))]) if len(y) else np.zeros(T, dtype=int)
        return Xp[None].astype('float32'), np.array([assign_window_label(yp, label_rule, benign_label)]), np.array([0])
    for start in range(0, len(X) - T + 1, stride):
        end = start + T
        windows.append(X[start:end])
        labels.append(assign_window_label(y[start:end], label_rule, benign_label))
        starts.append(start)
    return np.stack(windows).astype('float32'), np.asarray(labels), np.asarray(starts)
