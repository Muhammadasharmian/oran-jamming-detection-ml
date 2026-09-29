# O-RAN Jamming Detection: Machine Learning Models

Training and inference code for detecting RF jamming in 5G NR / O-RAN networks from gNB Key Performance Metrics (KPMs). The package implements classical, gradient-boosting, deep learning, deep reinforcement learning and ensemble detectors behind one common interface, together with a parser for srsRAN gNB console logs.

## Models

| Name | Type | Task |
|---|---|---|
| `random_forest` | Random Forest | multi-class |
| `svm` | Support Vector Machine (RBF) | multi-class |
| `isolation_forest` | Isolation Forest, fitted on normal traffic | binary |
| `catboost` | CatBoost | multi-class |
| `lightgbm` | LightGBM | multi-class |
| `extra_trees` | Extremely Randomized Trees | multi-class |
| `mlp` | Feed-forward network (64-32) | multi-class |
| `deep_mlp` | Feed-forward network (128-64-32, batch norm) | multi-class |
| `cnn_lstm` | Conv1D + LSTM | multi-class |
| `bi_cnn_lstm` | Two-stage Conv1D + 2-layer bidirectional LSTM | multi-class |
| `ddpg` | DDPG agent (MLP, LLM or hybrid actor) | binary |
| `ensemble_boosting` | CatBoost + LightGBM + ExtraTrees, weighted soft voting | multi-class |
| `ensemble_classical` | Random Forest + SVM + Isolation Forest | binary |
| `ensemble_catboost_iforest` | CatBoost + Isolation Forest | binary |

Every model is wrapped in a scikit-learn `Pipeline` (standard scaling followed by the estimator), so all of them expose `fit`, `predict` and `predict_proba`. Binary models predict `normal` or `jamming`; the training script converts labels automatically when one is selected.

Ensemble weights start uniform and are fitted on the validation split (together with the decision threshold for the binary ensembles). No weights, thresholds or scores are hard-coded.

## Installation

```bash
git clone https://github.com/Muhammadasharmian/oran-jamming-detection-ml.git
cd oran-jamming-detection-ml
pip install -e ".[all,dev]"
```

Python 3.9 or later. `optuna` (tuning), `imbalanced-learn` (oversampling) and `transformers` (DDPG `llm` / `hybrid` actors) are optional.

## Data

The loader accepts two layouts:

1. **One CSV per class.** The class is taken from the file name; files starting with `normal` map to `normal`. This matches the [srsRAN gNB KPM dataset](https://github.com/Muhammadasharmian/srsran-gnb-kpm-dataset):

   ```bash
   git clone https://github.com/Muhammadasharmian/srsran-gnb-kpm-dataset.git data/srsran-gnb-kpm-dataset
   ```

2. **A single CSV with a label column**, selected with `--label-column`.

Identifier columns (`pci`, `rnti`, `source_file`, `timestamp`) are dropped because they identify the capture session and would leak the label. Missing values (`n/a`, `ovl`) are filled with zero and constant columns are removed.

Raw srsRAN gNB console captures (plain text or binary `script` recordings) can be converted with:

```bash
python scripts/parse_srsran_logs.py \
    --group normal="raw/normal_*.txt" \
    --group power_jamming="raw/power_*.txt" \
    --output-dir data/kpm
```

## Usage

Train and evaluate one model. The data is split into stratified train / validation / test partitions (70 / 10 / 20 by default), and the fitted model is saved to `artifacts/<model>.joblib` with its metrics.

```bash
python scripts/train.py --model ensemble_boosting --data data/srsran-gnb-kpm-dataset
python scripts/train.py --model catboost --data data/srsran-gnb-kpm-dataset --binary --latency
python scripts/train.py --model ddpg --data data/srsran-gnb-kpm-dataset --set model__total_steps=20000
```

Useful options:

| Option | Description |
|---|---|
| `--binary` | Collapse all jamming classes into one `jamming` class |
| `--oversample` | Balance the training partition with SMOTE (never applied to validation or test data) |
| `--no-tune-ensemble` | Keep uniform ensemble weights |
| `--set KEY=VALUE` | Override estimator parameters, e.g. `model__iterations=500` or `model__actor_type=hybrid` |

Compare all models on the same split:

```bash
python scripts/benchmark.py --data data/srsran-gnb-kpm-dataset --output results/benchmark.csv
```

Search hyperparameters with Optuna (cross-validation on the training partition only):

```bash
python scripts/tune.py --model catboost --data data/srsran-gnb-kpm-dataset --trials 50
```

Run a trained model on new data (CSV or raw gNB log):

```bash
python scripts/predict.py --model artifacts/ensemble_boosting.joblib --input new_capture.csv --output predictions.csv
```

Reported metrics: accuracy, macro and weighted F1, MCC, per-class report, confusion matrix, and binary jamming-vs-normal precision, recall, F1, false-alarm rate, miss rate and ROC AUC.

## Project structure

```
jamming_detection/
    config.py            Default hyperparameters for every model
    data/
        loader.py        CSV loading, labelling, splitting, SMOTE
        srsran_parser.py srsRAN gNB console log parser
    models/
        classical.py     Random Forest, SVM, Isolation Forest
        boosting.py      CatBoost, LightGBM, ExtraTrees
        neural.py        MLP and CNN-LSTM networks (PyTorch)
        drl.py           DDPG agent and actors
        ensemble.py      Weighted ensembles
    envs/dataset_env.py  Gymnasium environment for DDPG training
    evaluation.py        Metrics and latency measurement
    training.py          Training workflow and model persistence
scripts/                 train, benchmark, tune, predict, parse_srsran_logs
tests/                   Smoke tests for the parser, loader and every model
```

## Testing

```bash
pytest
```

The tests train every model briefly on a small synthetic dataset to check the fit, predict and save / load paths. They do not measure detection performance.

## Notes on the DDPG detector

The DDPG agent sees one KPM sample per step and outputs a jamming score in `[-1, 1]`. The reward is class-weighted and proportional to how close the score is to the true label. Because samples are independent, training uses a discount factor of zero. The `llm` and `hybrid` actors encode a text description of the state with a frozen DistilBERT model, which is downloaded on first use.
