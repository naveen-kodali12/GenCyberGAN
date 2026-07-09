## 📌 Software Archive

The official archived release of **GenCyberGAN Version 1.1** is available on Zenodo.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.21274578

# GenCyberGAN

## Graph-Temporal Cyber Threat Intelligence Conditioned Generative Adversarial Network for Synthetic Intrusion Data Generation and Intelligent Cyberattack Detection

GenCyberGAN is a PyTorch-based deep learning framework designed for generating realistic synthetic cyberattack traffic using a **Cyber Threat Intelligence (CTI)-conditioned Graph-Temporal Generative Adversarial Network (GAN)**. The generated samples are used to augment intrusion detection datasets and improve the performance of Intelligent Intrusion Detection Systems (IDS).

The framework integrates graph learning, temporal modeling, CTI knowledge, and adversarial learning into a unified pipeline for cybersecurity research.

---

# Features

- Graph-Temporal GAN architecture
- Cyber Threat Intelligence (CTI) guided sample generation
- Flow-level temporal window construction
- Feature harmonization and preprocessing
- Heterogeneous communication graph generation
- Multi-Critic WGAN-GP training
- Synthetic attack sample generation
- Mahalanobis distance-based filtering
- IDS training and evaluation
- Automated performance reporting and visualization

---

# Repository Structure

```
GenCyberGAN/
│
├── configs/                 # Configuration files
├── data/                    # Input datasets
├── experiments/             # Experimental outputs
├── results/                 # Generated results
│
├── src/
│   ├── models/              # Generator, Critics, IDS Models
│   ├── dataset_loader.py
│   ├── data_preprocessing.py
│   ├── feature_harmonizer.py
│   ├── temporal_windowing.py
│   ├── graph_builder.py
│   ├── cti_mapping.py
│   ├── losses.py
│   ├── train_gencybergan.py
│   ├── generate_synthetic.py
│   ├── train_ids.py
│   ├── evaluate.py
│   └── utils.py
│
├── main.py
├── requirements.txt
```

---

# Workflow

The complete GenCyberGAN pipeline consists of the following stages:

1. Dataset Loading
2. Data Cleaning and Feature Harmonization
3. Numerical Scaling and Encoding
4. Temporal Window Construction
5. Communication Graph Generation
6. CTI Knowledge Mapping
7. Graph-Temporal GAN Training
8. Synthetic Sample Generation
9. Synthetic Sample Filtering
10. IDS Model Training
11. Performance Evaluation

---

# Installation

Clone the repository

```bash
git clone https://github.com/naveen-kodali12/GenCyberGAN
```

Navigate into the project directory

```bash
cd GenCyberGAN
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Complete Pipeline

Execute the entire framework using

```bash
python main.py --mode full
```

---

# Individual Modules

### Data Preprocessing

```bash
python main.py --mode preprocess
```

### Train GenCyberGAN

```bash
python main.py --mode train_gan
```

### Generate Synthetic Samples

```bash
python main.py --mode generate
```

### Train Intrusion Detection System

```bash
python main.py --mode train_ids
```

---

# Using Your Own Dataset

Example:

```bash
python main.py --dataset unsw \
--raw_path /path/to/UNSW_NB15.csv \
--mode full \
--output_dir results
```

The framework automatically harmonizes common network traffic features including:

- Source IP
- Destination IP
- Protocol
- Service
- Duration
- Source Bytes
- Destination Bytes
- Attack Category
- Labels

---

# Output Files

The framework generates:

- Processed datasets
- CTI vocabulary
- Label mappings
- Trained Generator
- Flow Critic
- Graph Critic
- Synthetic samples
- Filtered synthetic samples
- IDS models
- Classification reports
- Confusion matrices
- Performance metrics
- CSV result files

---

# Supported Datasets

The framework can be adapted for several cybersecurity datasets, including:

- CICIDS2017
- UNSW-NB15
- Bot-IoT
- CSE-CIC-IDS2018
- TON_IoT
- Other flow-based intrusion detection datasets

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- NetworkX
- Matplotlib
- YAML

---

# Research Applications

GenCyberGAN can be applied to:

- Intrusion Detection Systems
- Cyber Threat Intelligence
- Network Security Analytics
- Synthetic Cyberattack Generation
- Security Data Augmentation
- AI-driven Cybersecurity
- Intelligent Threat Detection

---

# Citation

If you use this repository in your research, please cite the associated publication.

```text
Author(s),
Graph-Temporal Cyber Threat Intelligence Conditioned Generative Adversarial Network for Synthetic Intrusion Data Generation and Intelligent Cyberattack Detection,
Year.
```

---

# License

This project is released for academic and research purposes.

---

