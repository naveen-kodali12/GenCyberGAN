# GenCyberGAN

Functional PyTorch research prototype for **CTI-conditioned graph-temporal generation of cyberattack flow windows** and IDS augmentation.

The code implements the complete planned pipeline:

1. Dataset loading and feature harmonization
2. Cleaning, label harmonization, numerical scaling and categorical encoding
3. Temporal sliding-window construction
4. Heterogeneous communication graph descriptor extraction
5. CTI tactic-technique-service mapping
6. CTI-conditioned graph-temporal WGAN-GP generator
7. Multi-critic training with flow critic, graph critic and CTI classifier
8. Mahalanobis filtering of generated graph descriptors
9. Downstream IDS training and evaluation
10. Result CSV files, classification reports and confusion matrix plots

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick functional demo

The demo mode creates a small synthetic IDS-like flow dataset, so the repository runs without downloading external data.

```bash
python main.py --mode full
```

Outputs are written to `results/`.

## Use with a real CSV IDS dataset

```bash
python main.py --dataset unsw --raw_path /path/to/UNSW_NB15.csv --mode full --output_dir results_unsw
```

The loader automatically tries to harmonize common columns such as `srcip`, `dstip`, `proto`, `service`, `state`, `dur`, `sbytes`, `dbytes`, `attack_cat`, and `label`.

## Individual stages

```bash
python main.py --mode preprocess
python main.py --mode train_gan
python main.py --mode generate
python main.py --mode train_ids
```

## Main outputs

- `processed_train.npz`
- `processed_val.npz`
- `processed_test.npz`
- `cti_vocab.json`
- `label_mapping.json`
- `gencybergan_generator.pt`
- `flow_critic.pt`
- `graph_critic.pt`
- `synthetic_samples.npz`
- `synthetic_samples_filtered.npz`
- `ids_baseline.pt`
- `ids_gencybergan.pt`
- `results_baseline.csv`
- `results_gencybergan.csv`
- `classification_report_*.txt`
- `confusion_matrix_*.png`

## Notes for paper experiments

For full SCI-level experiments, run separately on CICIDS2017, UNSW-NB15 and Bot-IoT after setting suitable CSV paths. Increase `gan_epochs` and `ids_epochs` in `configs/default.yaml` to 50-100 for real experiments. The default uses small epochs so the pipeline can be verified quickly.
