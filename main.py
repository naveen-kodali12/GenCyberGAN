import os, argparse
import numpy as np
from src.utils import load_config, set_seed, get_device, ensure_dir, save_npz, save_json
from src.data_preprocessing import load_dataset, prepare_dataframe, split_dataframe, fit_transform_features
from src.temporal_windowing import create_temporal_windows
from src.graph_builder import descriptors_from_windows
from src.cti_mapping import build_cti_vocab, encode_labels_to_cti_ids, save_cti_vocab
from src.train_gencybergan import train_gencybergan
from src.generate_synthetic import generate_synthetic_samples
from src.train_ids import train_ids_model
from src.evaluate import evaluate_ids


def preprocess(cfg):
    out = cfg['output_dir']; ensure_dir(out)
    dcfg = cfg['data']
    df = load_dataset(dcfg.get('raw_path',''), dcfg.get('dataset_name','demo'), dcfg.get('max_rows'))
    df = prepare_dataframe(df, dcfg.get('label_col','label'), dcfg.get('timestamp_col','timestamp'))
    train_df, val_df, test_df = split_dataframe(df, 'label', dcfg.get('test_size',0.2), dcfg.get('val_size',0.1), cfg.get('seed',42), dcfg.get('unseen_families',[]))
    (trX, ytr), (vX, yv), (teX, yte), features, le, scaler, enc = fit_transform_features(train_df, val_df, test_df, 'label')
    T, stride = dcfg.get('window_length',30), dcfg.get('stride',5)
    Xtr, wytr, starts_tr = create_temporal_windows(trX, ytr, T, stride, dcfg.get('label_rule','any_attack'))
    Xv, wyv, starts_v = create_temporal_windows(vX, yv, T, stride, dcfg.get('label_rule','any_attack'))
    Xte, wyte, starts_te = create_temporal_windows(teX, yte, T, stride, dcfg.get('label_rule','any_attack'))
    gtr = descriptors_from_windows(train_df, starts_tr, T); gv = descriptors_from_windows(val_df, starts_v, T); gte = descriptors_from_windows(test_df, starts_te, T)
    labels_text_train = le.inverse_transform(wytr.clip(0, len(le.classes_)-1))
    labels_text_val = le.inverse_transform(wyv.clip(0, len(le.classes_)-1))
    labels_text_test = [le.classes_[i] if 0 <= i < len(le.classes_) else 'Unknown' for i in wyte]
    vocab = build_cti_vocab(list(labels_text_train) + list(labels_text_val))
    ctr = encode_labels_to_cti_ids(labels_text_train, vocab); cv = encode_labels_to_cti_ids(labels_text_val, vocab); cte = encode_labels_to_cti_ids(labels_text_test, vocab)
    save_npz(os.path.join(out,'processed_train.npz'), X=Xtr, graph_desc=gtr, y=wytr, cti_ids=ctr)
    save_npz(os.path.join(out,'processed_val.npz'), X=Xv, graph_desc=gv, y=wyv, cti_ids=cv)
    save_npz(os.path.join(out,'processed_test.npz'), X=Xte, graph_desc=gte, y=wyte, cti_ids=cte)
    save_cti_vocab(vocab, os.path.join(out,'cti_vocab.json'))
    save_json({'classes': le.classes_.tolist(), 'features': features}, os.path.join(out,'label_mapping.json'))
    print(f'Preprocessing complete. Train windows: {Xtr.shape}, Val: {Xv.shape}, Test: {Xte.shape}')


def load_processed(out):
    tr = np.load(os.path.join(out,'processed_train.npz'), allow_pickle=True)
    va = np.load(os.path.join(out,'processed_val.npz'), allow_pickle=True)
    te = np.load(os.path.join(out,'processed_test.npz'), allow_pickle=True)
    return tr, va, te


def run_train_gan(cfg, device):
    out = cfg['output_dir']; tr, _, _ = load_processed(out)
    cti = tr['cti_ids']; vocab_sizes = [int(cti[:,i].max()+1) if len(cti) else 1 for i in range(3)]
    return train_gencybergan(tr['X'], tr['graph_desc'], tr['cti_ids'], tr['y'], vocab_sizes, cfg, out, device)


def run_generate(cfg, device, generator=None):
    out = cfg['output_dir']; tr, _, _ = load_processed(out)
    if generator is None:
        import torch
        from src.models.generator import GenCyberGenerator
        ckpt = torch.load(os.path.join(out,'gencybergan_generator.pt'), map_location=device)
        m = cfg['model']; generator = GenCyberGenerator(ckpt['vocab_sizes'], m['z_dim'], m['cti_emb_dim'], m['cti_dim'], m['hidden_dim'], ckpt['T'], ckpt['feature_dim'], ckpt['graph_desc_dim'], m.get('dropout',0.1)).to(device)
        generator.load_state_dict(ckpt['model'])
    return generate_synthetic_samples(generator, tr['cti_ids'], tr['y'], cfg, out, device, tr['graph_desc'])


def run_train_ids(cfg, device):
    out = cfg['output_dir']; tr, va, te = load_processed(out)
    baseline = train_ids_model(tr['X'], tr['y'], va['X'], va['y'], cfg, out, device, 'baseline')
    synth_path = os.path.join(out,'synthetic_samples_filtered.npz')
    if os.path.exists(synth_path):
        syn = np.load(synth_path, allow_pickle=True)
        Xaug = np.concatenate([tr['X'], syn['X']], axis=0)
        yaug = np.concatenate([tr['y'], syn['y']], axis=0)
        gen = train_ids_model(Xaug, yaug, va['X'], va['y'], cfg, out, device, 'gencybergan')
    else:
        gen = None
    res1 = evaluate_ids(baseline, te['X'], te['y'], out, tag='baseline', device=device)
    print('Baseline IDS:', res1)
    if gen:
        res2 = evaluate_ids(gen, te['X'], te['y'], out, tag='gencybergan', device=device)
        print('GenCyberGAN IDS:', res2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--mode', choices=['preprocess','train_gan','generate','train_ids','evaluate','full'], default='full')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--raw_path', default=None)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset: cfg['data']['dataset_name'] = args.dataset
    if args.raw_path: cfg['data']['raw_path'] = args.raw_path
    if args.output_dir: cfg['output_dir'] = args.output_dir
    set_seed(cfg.get('seed',42)); ensure_dir(cfg['output_dir'])
    device = get_device(cfg['training'].get('device','auto'))
    print('Using device:', device)
    if args.mode in ['preprocess','full']:
        preprocess(cfg)
    G = None
    if args.mode in ['train_gan','full']:
        G = run_train_gan(cfg, device)
    if args.mode in ['generate','full']:
        run_generate(cfg, device, G)
    if args.mode in ['train_ids','evaluate','full']:
        run_train_ids(cfg, device)

if __name__ == '__main__':
    main()
