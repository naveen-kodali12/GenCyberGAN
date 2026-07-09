import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from .feature_harmonizer import harmonize_column_names, harmonize_attack_label


def make_demo_dataset(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(['Benign','DoS_DDoS','Reconnaissance','BruteForce','Botnet','RareExploit'], n,
                        p=[0.58,0.17,0.10,0.08,0.05,0.02])
    t = np.arange(n)
    df = pd.DataFrame({
        'timestamp': t,
        'src_ip': [f'10.0.{rng.integers(0,8)}.{rng.integers(1,255)}' for _ in range(n)],
        'dst_ip': [f'172.16.{rng.integers(0,8)}.{rng.integers(1,255)}' for _ in range(n)],
        'src_port': rng.integers(1024, 65000, n),
        'dst_port': rng.choice([22,53,80,443,445,3389,8080], n),
        'proto': rng.choice(['tcp','udp','icmp'], n, p=[0.68,0.27,0.05]),
        'service': rng.choice(['http','dns','ssh','ftp','ssl','smtp','unknown'], n),
        'state': rng.choice(['FIN','CON','INT','REQ','RST'], n),
        'duration': rng.exponential(1.2, n),
        'bytes': rng.lognormal(6, 1.2, n),
        'packets': rng.poisson(12, n) + 1,
        'iat_mean': rng.exponential(0.3, n),
        'flow_rate': rng.lognormal(2, 0.8, n),
        'label': labels
    })
    attack_mask = df['label'] != 'Benign'
    df.loc[attack_mask, 'bytes'] *= rng.uniform(1.4, 4.5, attack_mask.sum())
    df.loc[attack_mask, 'packets'] += rng.poisson(20, attack_mask.sum())
    return df


def load_dataset(path='', dataset_name='demo', max_rows=None):
    if not path or dataset_name.lower() == 'demo':
        df = make_demo_dataset(n=max_rows or 2500)
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Raw dataset not found: {path}')
        df = pd.read_csv(path, low_memory=False)
        if max_rows:
            df = df.head(max_rows)
    return harmonize_column_names(df)


def clean_dataframe(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop_duplicates(inplace=True)
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna('unknown').astype(str)
    return df.reset_index(drop=True)


def prepare_dataframe(df, label_col='label', timestamp_col='timestamp'):
    df = clean_dataframe(df)
    if label_col not in df.columns:
        raise ValueError(f'Missing label column {label_col}. Available columns: {list(df.columns)[:20]}')
    df[label_col] = df[label_col].map(harmonize_attack_label)
    if timestamp_col not in df.columns:
        df[timestamp_col] = np.arange(len(df))
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def split_dataframe(df, label_col='label', test_size=0.2, val_size=0.1, seed=42, unseen_families=None):
    unseen_families = unseen_families or []
    unseen_df = df[df[label_col].isin(unseen_families)].copy()
    seen_df = df[~df[label_col].isin(unseen_families)].copy()
    vc = seen_df[label_col].value_counts()
    strat = seen_df[label_col] if seen_df[label_col].nunique() > 1 and vc.min() >= 2 else None
    train_val, test_seen = train_test_split(seen_df, test_size=test_size, random_state=seed, stratify=strat)
    vc_tv = train_val[label_col].value_counts()
    strat_tv = train_val[label_col] if train_val[label_col].nunique() > 1 and vc_tv.min() >= 2 else None
    val_ratio = val_size / max(1e-9, (1 - test_size))
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed, stratify=strat_tv)
    test = pd.concat([test_seen, unseen_df], axis=0).sort_index().reset_index(drop=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def fit_transform_features(train, val, test, label_col='label', exclude_cols=None):
    exclude_cols = set(exclude_cols or []) | {label_col}
    meta_cols = [c for c in ['timestamp','src_ip','dst_ip','src_port','dst_port','proto','service','state'] if c in train.columns]
    num_cols = [c for c in train.columns if c not in exclude_cols and c not in meta_cols and train[c].dtype.kind in 'biufc']
    cat_cols = [c for c in ['proto','service','state'] if c in train.columns]
    scaler = StandardScaler()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    for part in [train, val, test]:
        for c in num_cols:
            part[c] = pd.to_numeric(part[c], errors='coerce').fillna(0)
        for c in cat_cols:
            part[c] = part[c].astype(str).fillna('unknown')
    Xtr_num = scaler.fit_transform(train[num_cols]) if num_cols else np.zeros((len(train),0))
    Xv_num = scaler.transform(val[num_cols]) if num_cols else np.zeros((len(val),0))
    Xte_num = scaler.transform(test[num_cols]) if num_cols else np.zeros((len(test),0))
    Xtr_cat = enc.fit_transform(train[cat_cols]) if cat_cols else np.zeros((len(train),0))
    Xv_cat = enc.transform(val[cat_cols]) if cat_cols else np.zeros((len(val),0))
    Xte_cat = enc.transform(test[cat_cols]) if cat_cols else np.zeros((len(test),0))
    # scale ordinal categorical values to small numeric range
    denom = np.maximum(1, np.array([len(cats)-1 for cats in enc.categories_], dtype=float)) if cat_cols else np.array([])
    if cat_cols:
        Xtr_cat = np.where(Xtr_cat < 0, 0, Xtr_cat) / denom
        Xv_cat = np.where(Xv_cat < 0, 0, Xv_cat) / denom
        Xte_cat = np.where(Xte_cat < 0, 0, Xte_cat) / denom
    features = num_cols + cat_cols
    train_X = np.hstack([Xtr_num, Xtr_cat]).astype('float32')
    val_X = np.hstack([Xv_num, Xv_cat]).astype('float32')
    test_X = np.hstack([Xte_num, Xte_cat]).astype('float32')
    le = LabelEncoder()
    y_train = le.fit_transform(train[label_col])
    y_val = le.transform(val[label_col].where(val[label_col].isin(le.classes_), le.classes_[0]))
    # unseen labels may not be in train; add mapping manually later as -1 fallback
    class_to_idx = {c:i for i,c in enumerate(le.classes_)}
    y_test = np.array([class_to_idx.get(x, -1) for x in test[label_col]], dtype=int)
    return (train_X, y_train), (val_X, y_val), (test_X, y_test), features, le, scaler, enc
