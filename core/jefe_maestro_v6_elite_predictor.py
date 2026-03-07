#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

jefe_maestro_v6_elite_predictor.py

Version: v6.0 Elite Predictor (A4)

Resumen:

- Pipeline híbrido en 3 fases: muestreo por importancia (prerank) -> expansión de vecindarios -> evaluación final.

- Global composite: una sola combinación participa en Melate/Revancha/Revanchita.

- No se generan datos mock bajo ninguna circunstancia. Si faltan datos reales el programa ABORTA y alerta.

- Ensamble ML interno con pesos personalizados (ELITE):

    IF  = 0.15

    XGB = 0.45

    LGBM= 0.20

    LSTM= 0.20

- Estadística adicional (hot scores, entropy, parity, sum balance) se usan como refinamiento.

- Multiprocessing Windows-safe (get_context("spawn")), initializer top-level.

- Logs, alertas por Telegram y Email, guardado de resultados y historial de predicciones.

- Diseñado para correr eficientemente en una máquina de 4 núcleos.

"""

from __future__ import annotations

import os

import sys

import math

import json

import time

import gc

import logging

import warnings

import argparse

import itertools

import random

from datetime import datetime, timedelta

from functools import lru_cache

from typing import List, Tuple, Dict, Any

# Reduce TF verbosity

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ---------- Imports ----------

import numpy as np

import pandas as pd

import joblib

from sklearn.ensemble import IsolationForest

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.metrics import make_scorer, roc_auc_score

import xgboost as xgb

import lightgbm as lgb

try:

    import tensorflow as tf

    tf.get_logger().setLevel('ERROR')

    from tensorflow.keras.models import Sequential

    from tensorflow.keras.layers import LSTM, Dense

    TENSORFLOW_AVAILABLE = True

except Exception:

    TENSORFLOW_AVAILABLE = False

import psutil

from imblearn.over_sampling import SMOTE, RandomOverSampler

import requests

import smtplib

from email.mime.multipart import MIMEMultipart

from email.mime.text import MIMEText

from multiprocessing import get_context, cpu_count

from logging.handlers import RotatingFileHandler

# ---------- Logging ----------

warnings.filterwarnings("ignore")

LOG_FILE = "jefe_maestro_v6_elite.log"

handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")

stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s - %(levelname)s - %(message)s",

    handlers=[handler, stream_handler]

)

logger = logging.getLogger(__name__)

# ---------- Configuration ----------

LOTTERIES = {

    "Melate": {"n_max": 56, "k": 6, "sheet_key": os.getenv("MELATE_SHEET_KEY", "1C3tA9spKhsJB9sTSwTb3Z2nY_sxLD05EZDZ7SrKuJz4")},

    "Revancha": {"n_max": 56, "k": 6, "sheet_key": os.getenv("REVANCHA_SHEET_KEY", "1NHmzbhCt4xroSZYPYkVs--IA-yPM7N9WaN36-kAiCLs")},

    "Revanchita": {"n_max": 56, "k": 6, "sheet_key": os.getenv("REVANCHITA_SHEET_KEY", "1G8o0_DtaQl9JjoDCVtHhNfIfJQXXauBY6MhWPHSD1sU")},

}

MODEL_FILE_TEMPLATE = "model_{name}.joblib"

PREDICTIONS_HISTORY_FILE = "predictions_history.json"

HDF_CACHE = "cache/histories.h5"

# Elite ensemble weights (A4 - Elite Predictor)

ALPHA_IF = 0.15

BETA_XGB = 0.45

BETA_LGBM = 0.20

# LSTM weight computed as remainder to ensure ML ensemble sums to 1.0

# LSTM weight = 1.0 - (ALPHA_IF + BETA_XGB + BETA_LGBM) = 0.20

# Statistical refinement weights (small)

GAMMA_HOT = 0.06

DELTA_ENT = 0.06

EPS_PAR = 0.04

ZETA_SUM = 0.04

ALPHA_DECAY = 0.95

SEED = int(os.getenv("JM_SEED", "42"))

np.random.seed(SEED)

random.seed(SEED)

TOP_K = int(os.getenv("JM_TOP_K", "20"))

WORKERS = int(os.getenv("JM_WORKERS", str(max(1, min(4, cpu_count())))))  # prioritize up to 4 cores

BATCH_SIZE = int(os.getenv("JM_BATCH_SIZE", "120000"))

# Candidate sampling & expansion targets (tunable)

PRERANK_SAMPLES = int(os.getenv("PRERANK_SAMPLES", "160000"))      # initial importance-sampling

PRERANK_TOP = int(os.getenv("PRERANK_TOP", "50000"))              # keep top N to expand

NEIGHBORS_PER_COMBO = int(os.getenv("NEIGHBORS_PER_COMBO", "20")) # neighbors per top combo

TARGET_EVAL = int(os.getenv("TARGET_EVAL", "2000000"))            # target final evaluations (<= ~5M recommended)

FINAL_BATCH = int(os.getenv("FINAL_BATCH", "200000"))             # batch size for final evals

# Plausibility filters (safe ranges)

MIN_SUM_ALLOWED = 60

MAX_SUM_ALLOWED = 300

MAX_CONSECUTIVE_ALLOWED = 4

# Email & Telegram

EMAIL_FROM = os.getenv("EMAIL_USER")

EMAIL_PASS = os.getenv("EMAIL_PASS")

EMAIL_TO = os.getenv("EMAIL_TO") or EMAIL_FROM

SMTP_SERVER = "smtp.gmail.com"

SMTP_PORT = 587

# Directories

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(os.path.dirname(HDF_CACHE), exist_ok=True)

# Globals for worker processes

WORKER_MODELS: Dict[str, Dict[str, Any]] = {}

WORKER_SCALERS: Dict[str, Any] = {}

WORKER_NAMES: List[str] = []

# ---------- Utilities & Alerts ----------

def safe_json_convert(obj):

    if isinstance(obj, (np.integer,)):

        return int(obj)

    if isinstance(obj, (np.floating,)):

        return float(obj)

    if isinstance(obj, (np.ndarray,)):

        return obj.tolist()

    if isinstance(obj, (pd.Timestamp,)):

        return obj.isoformat()

    if isinstance(obj, (pd.Series,)):

        return obj.tolist()

    return obj

def send_telegram_alert(message: str):

    token = os.getenv("TELEGRAM_TOKEN")

    chat = os.getenv("TELEGRAM_CHAT_ID")

    if token and chat:

        try:

            url = f"https://api.telegram.org/bot{token}/sendMessage"

            payload = {"chat_id": chat, "text": message}

            requests.post(url, json=payload, timeout=10)

            logger.info("Telegram alert sent")

        except Exception as e:

            logger.error(f"Error sending Telegram alert: {e}")

def send_email_alert(subject: str, body: str):

    if not EMAIL_FROM or not EMAIL_PASS or not EMAIL_TO:

        logger.warning("Email credentials not set - skipping alert email")

        return False

    msg = MIMEMultipart()

    msg["Subject"] = subject

    msg["From"] = EMAIL_FROM

    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(body, "plain"))

    try:

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:

            server.ehlo()

            server.starttls()

            server.login(EMAIL_FROM, EMAIL_PASS)

            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())

        logger.info("Alert email sent")

        return True

    except Exception as e:

        logger.error(f"Failed to send alert email: {e}")

        return False

def abort_no_data(reason: str):

    logger.critical(f"ABORT: {reason}")

    send_telegram_alert(f"Jefe Maestro - ABORT: {reason}")

    send_email_alert("Jefe Maestro - ABORT: datos insuficientes", f"Se abortó la ejecución por: {reason}\nRevisa logs en {LOG_FILE}")

    sys.exit(2)

# ---------- Google Sheets loader (strict) ----------

# ---------- Local CSV loader (strict) ----------

def load_history_strict(name: str) -> pd.DataFrame:

    try:

        file_map = {
            "Melate": "data/melate.csv",
            "Revancha": "data/revancha.csv",
            "Revanchita": "data/revanchita.csv"
        }

        path = file_map.get(name)

        if not path or not os.path.exists(path):
            raise RuntimeError(f"Archivo CSV no encontrado para {name}: {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise RuntimeError(f"CSV vacío para {name}")

        df["FUENTE"] = name

        if "FECHA" in df.columns:
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

        k = LOTTERIES[name]["k"]

        expected_cols = [f"N{i}" for i in range(1, k + 1)]

        if not all(col in df.columns for col in expected_cols):
            raise RuntimeError(f"CSV {name} falta columnas esperadas: {expected_cols}")

        df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors="coerce")

        df = df.dropna(subset=expected_cols)

        if df.empty:
            raise RuntimeError(f"CSV {name} no contiene filas válidas")

        n_max = LOTTERIES[name]["n_max"]

        for col in expected_cols:
            if not df[col].apply(lambda x: 1 <= int(x) <= n_max).all():
                raise RuntimeError(f"CSV {name} tiene valores fuera de rango en {col}")

        return df.sort_values("FECHA").reset_index(drop=True)

    except Exception as e:

        logger.error(f"Error leyendo CSV {name}: {e}")
        raise


def load_all_histories_strict() -> Dict[str, pd.DataFrame]:

    dfs = {}

    for name in LOTTERIES.keys():

        try:

            df = load_history_strict(name)
            dfs[name] = df

            logger.info(f"[{name}] Historial cargado desde CSV: {len(df):,} filas")

        except Exception as e:

            abort_no_data(f"No se pudo cargar el historial para {name}: {e}")

    return dfs

# ---------- Feature engineering ----------

@lru_cache(maxsize=400000)

def combo_basic_features_tuple(nums: Tuple[int, ...], n_max: int) -> Tuple[float, ...]:

    nums = tuple(sorted(int(x) for x in nums))

    s = float(sum(nums))

    uniq = float(len(set(nums)))

    evens = float(sum(1 for n in nums if n % 2 == 0))

    rng = float(max(nums) - min(nums)) if nums else 0.0

    last_digits = [n % 10 for n in nums]

    counts = {}

    for d in last_digits:

        counts[d] = counts.get(d, 0) + 1

    total_ld = sum(counts.values())

    entropy = 0.0

    for v in counts.values():

        p = v / (total_ld + 1e-12)

        entropy -= p * math.log2(p + 1e-12)

    consecutive = float(sum(1 for i in range(len(nums)-1) if nums[i+1] == nums[i] + 1))

    std = float(np.std(nums))

    diffs = np.diff(nums) if len(nums) > 1 else np.array([0])

    min_dist = float(diffs.min()) if diffs.size > 0 else 0.0

    max_dist = float(diffs.max()) if diffs.size > 0 else 0.0

    bins = np.histogram(nums, bins=max(1, math.ceil(n_max/10)), range=(1, n_max+1))[0]

    decenas_norm = bins / (bins.sum() + 1e-9)

    decenas_entropy = 0.0

    for p in decenas_norm:

        if p > 0:

            decenas_entropy -= p * math.log2(p + 1e-12)

    return (s, uniq, evens, rng, entropy, consecutive, std, min_dist, max_dist, decenas_entropy)

def build_hot_freq(df_hist: pd.DataFrame, last_k: int = 50, k: int = 6) -> Dict[str, float]:

    tail = df_hist.tail(last_k)

    if tail.empty:

        return {}

    freq: Dict[int, float] = {}

    for idx, (_, row) in enumerate(reversed(list(tail.iterrows()))):

        w = ALPHA_DECAY ** idx

        for i in range(1, k+1):

            col = f"N{i}"

            if col in row and pd.notna(row[col]):

                n = int(row[col])

                freq[n] = freq.get(n, 0.0) + w

    total = sum(freq.values())

    if total > 0:

        for n in list(freq.keys()):

            freq[n] /= total

    return {str(k): float(v) for k, v in freq.items()}

def build_pos_freq(df_hist: pd.DataFrame, k: int) -> List[Dict[str, float]]:

    pos_freq = [{} for _ in range(k)]

    n_rows = len(df_hist)

    for idx, (_, row) in enumerate(df_hist.iterrows()):

        w = ALPHA_DECAY ** (n_rows - idx - 1)

        for i in range(k):

            col = f"N{i+1}"

            if col in row and pd.notna(row[col]):

                n = int(row[col])

                pos_freq[i][n] = pos_freq[i].get(n, 0.0) + w

    for p in pos_freq:

        total = sum(p.values())

        if total > 0:

            for n in list(p.keys()):

                p[n] /= total

    return [{str(k): float(v) for k, v in p.items()} for p in pos_freq]

@lru_cache(maxsize=400000)

def enhanced_combo_features_tuple(nums: Tuple[int, ...], hot_freq_json: str, pos_freq_jsons: Tuple[str, ...], n_max: int) -> Tuple[float, ...]:

    base = combo_basic_features_tuple(nums, n_max)

    try:

        hot_freq = json.loads(hot_freq_json) if hot_freq_json else {}

    except Exception:

        hot_freq = {}

    try:

        pos_list = [json.loads(p) for p in pos_freq_jsons] if pos_freq_jsons else [{} for _ in range(len(nums))]

    except Exception:

        pos_list = [{} for _ in range(len(nums))]

    hot_score = sum(float(hot_freq.get(str(n), 0.0)) for n in nums)

    pos_score_vals = [float(pos_list[i].get(str(nums[i]), 0.0)) if i < len(pos_list) else 0.0 for i in range(len(nums))]

    pos_score = float(np.mean(pos_score_vals)) if pos_score_vals else 0.0

    return tuple(list(base) + [hot_score, pos_score])

# ---------- Model training/loading ----------

def load_predictions_history():

    try:

        if os.path.exists(PREDICTIONS_HISTORY_FILE):

            with open(PREDICTIONS_HISTORY_FILE, 'r', encoding='utf-8') as f:

                return json.load(f)

        return {}

    except Exception as e:

        logger.error(f"Error loading predictions history: {e}")

        return {}

def tune_model(model, X: np.ndarray, y: np.ndarray, param_grid: Dict, is_if: bool = False):

    try:

        n_splits = max(2, min(4, max(2, len(X)//10)))

        tscv = TimeSeriesSplit(n_splits=n_splits)

        scorer = make_scorer(roc_auc_score, greater_is_better=not is_if)

        gs = GridSearchCV(model, param_grid, cv=tscv, scoring=scorer, n_jobs=1, error_score=0)

        gs.fit(X, y)

        logger.info(f"Best params: {gs.best_params_}, score: {gs.best_score_:.3f}")

        return gs.best_estimator_

    except Exception as e:

        try:

            logger.warning(f"Tuning failed ({e}), fitting default model")

            return model.fit(X, y)

        except Exception as e2:

            logger.error(f"Model fit failed: {e2}")

            return model

def build_supervised_dataset(df_hist: pd.DataFrame, name: str, n_neg: int = 2000):

    k = LOTTERIES[name]["k"]

    hot = build_hot_freq(df_hist, k=k)

    pos_freq = build_pos_freq(df_hist, k)

    n_max = LOTTERIES[name]["n_max"]

    history = load_predictions_history()

    feats, labels = [], []

    n_pos = 0

    for _, row in df_hist.iterrows():

        try:

            nums = tuple(sorted(int(row[f"N{i}"]) for i in range(1, k+1)))

        except Exception:

            continue

        feats.append(enhanced_combo_features_tuple(nums, json.dumps(hot), tuple(json.dumps(p) for p in pos_freq), n_max))

        labels.append(1)

        n_pos += 1

    if n_pos == 0:

        raise RuntimeError(f"No positive samples for {name}; cannot train")

    n_neg = min(n_neg, max(1, n_pos * 2))

    rng = np.random.default_rng(SEED)

    for _ in range(n_neg):

        nums = tuple(sorted(int(x) for x in rng.choice(range(1, n_max+1), k, replace=False)))

        feats.append(enhanced_combo_features_tuple(nums, json.dumps(hot), tuple(json.dumps(p) for p in pos_freq), n_max))

        labels.append(0)

    # incorporate previous predictions as weak supervision

    if name in history:

        for entry in history[name]:

            if entry.get("actual"):

                actual_set = set(entry["actual"])

                for pred in entry.get("predicted", []):

                    pred_combo = pred.get("combo")

                    if not pred_combo:

                        continue

                    pred_set = set(pred_combo)

                    match = len(pred_set & actual_set)

                    nums = tuple(sorted(int(x) for x in pred_combo))

                    feats.append(enhanced_combo_features_tuple(nums, json.dumps(hot), tuple(json.dumps(p) for p in pos_freq), n_max))

                    labels.append(1 if match >= 3 else 0)

    feats = np.array(feats, dtype=float)

    labels = np.array(labels, dtype=int)

    if len(labels) > 0 and len(np.unique(labels)) > 1:

        try:

            smote = SMOTE(random_state=SEED, k_neighbors=min(3, max(1, int(np.sum(labels==1))-1)))

            feats, labels = smote.fit_resample(feats, labels)

        except Exception as e:

            logger.warning(f"SMOTE failed, falling back to RandomOverSampler: {e}")

            ros = RandomOverSampler(random_state=SEED)

            feats, labels = ros.fit_resample(feats, labels)

    meta = {"hot": hot, "pos_freq": pos_freq}

    return feats, labels, meta

def validate_dataset_integrity(df: pd.DataFrame, name: str, n_max: int, k: int) -> bool:

    try:

        expected_cols = [f"N{i}" for i in range(1, k + 1)]

        if not all(col in df.columns for col in expected_cols):

            logger.error(f"[{name}] Missing columns: {set(expected_cols) - set(df.columns)}")

            return False

        for col in expected_cols:

            if not df[col].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)) and 1 <= int(x) <= n_max).all():

                logger.error(f"[{name}] Invalid values in {col}")

                return False

        return True

    except Exception as e:

        logger.error(f"[{name}] Error validating dataset: {e}")

        return False

def load_or_train_model(df_hist: pd.DataFrame, name: str) -> Dict[str, Any]:

    model_file = MODEL_FILE_TEMPLATE.format(name=name)

    if os.path.exists(model_file):

        try:

            models = joblib.load(model_file)

            logger.info(f"[{name}] Loaded model from {model_file}")

            return models

        except Exception as e:

            logger.warning(f"Error loading model {name}: {e} - will retrain")

    start_time = time.time()

    logger.info(f"[{name}] Training models...")

    if not validate_dataset_integrity(df_hist, name, LOTTERIES[name]["n_max"], LOTTERIES[name]["k"]):

        raise RuntimeError(f"Dataset invalid for {name}")

    X, y, meta = build_supervised_dataset(df_hist, name)

    if len(y) == 0:

        raise RuntimeError(f"No data for training for {name}")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Isolation Forest (anomaly)

    if_model = tune_model(IsolationForest(random_state=SEED), X_scaled, y, {'n_estimators': [100], 'contamination': [0.05]}, is_if=True)

    # XGBoost

    xgb_model = tune_model(xgb.XGBClassifier(random_state=SEED, eval_metric='auc', use_label_encoder=False), X_scaled, y,

                           {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.05], 'subsample': [0.8], 'colsample_bytree': [0.8]})

    # LightGBM

    try:

        lgbm_model = tune_model(lgb.LGBMClassifier(random_state=SEED, verbose=-1), X_scaled, y,

                                {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.05]})

    except Exception as e:

        logger.warning(f"LightGBM tuning failed: {e}")

        lgbm_model = None

    # LSTM (optional)

    lstm_model = None

    if TENSORFLOW_AVAILABLE and X_scaled.shape[0] > 80:

        try:

            lstm_model = Sequential([LSTM(32, input_shape=(X_scaled.shape[1], 1), return_sequences=False), Dense(1, activation='sigmoid')])

            X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

            lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

            lstm_model.fit(X_lstm, y, epochs=4, batch_size=32, validation_split=0.15, verbose=0)

        except Exception as e:

            logger.warning(f"LSTM training failed: {e}")

            lstm_model = None

    models = {'if': if_model, 'xgb': xgb_model, 'lgbm': lgbm_model, 'lstm': lstm_model, 'scaler': scaler, 'meta': meta}

    try:

        joblib.dump(models, model_file)

        logger.info(f"[{name}] Models saved to {model_file}")

    except Exception as e:

        logger.warning(f"Error saving model {name}: {e}")

        send_telegram_alert(f"Error saving model {name}: {e}")

    logger.info(f"[{name}] Models trained in {time.time() - start_time:.2f}s")

    return models

# ---------- Plausibility ----------

def is_plausible_combo(combo: Tuple[int, ...], min_sum=MIN_SUM_ALLOWED, max_sum=MAX_SUM_ALLOWED, max_consec=MAX_CONSECUTIVE_ALLOWED) -> bool:

    s = sum(combo)

    if s < min_sum or s > max_sum:

        return False

    consec = 1

    max_run = 1

    for i in range(len(combo)-1):

        if combo[i+1] == combo[i] + 1:

            consec += 1

            max_run = max(max_run, consec)

        else:

            consec = 1

    if max_run > max_consec:

        return False

    return True

# ---------- Worker initializer & scoring ----------

def worker_initializer(model_files_json: str, lottery_names_json: str):

    global WORKER_MODELS, WORKER_SCALERS, WORKER_NAMES

    try:

        model_files = json.loads(model_files_json)

        lottery_names = json.loads(lottery_names_json)

    except Exception:

        model_files = {}

        lottery_names = []

    WORKER_MODELS = {}

    WORKER_SCALERS = {}

    WORKER_NAMES = lottery_names

    for name in lottery_names:

        mf = model_files.get(name)

        if not mf or not os.path.exists(mf):

            WORKER_MODELS[name] = {}

            WORKER_SCALERS[name] = None

            logger.warning(f"[worker-{os.getpid()}] Modelo para {name} no encontrado: {mf}")

            continue

        try:

            models = joblib.load(mf)

            WORKER_MODELS[name] = {

                "if": models.get("if"),

                "xgb": models.get("xgb"),

                "lgbm": models.get("lgbm"),

                "lstm": models.get("lstm")

            }

            WORKER_SCALERS[name] = models.get("scaler")

            logger.info(f"[worker-{os.getpid()}] Modelos cargados para {name}")

        except Exception as e:

            WORKER_MODELS[name] = {}

            WORKER_SCALERS[name] = None

            logger.error(f"[worker-{os.getpid()}] Error cargando modelos para {name}: {e}")

    try:

        p = psutil.Process(os.getpid())

        if os.name == "nt":

            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

        else:

            p.nice(10)

    except Exception:

        pass

def _features_list_to_array(feats_list: List[Tuple[float, ...]]) -> np.ndarray:

    if not feats_list:

        return np.zeros((0, 0), dtype=np.float32)

    arr = np.array(feats_list, dtype=np.float32)

    return arr

def _norm_np(arr: np.ndarray) -> np.ndarray:

    if arr is None or arr.size == 0:

        return np.zeros_like(arr)

    a = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mn = a.min()

    mx = a.max()

    if mx - mn < 1e-12:

        return np.zeros_like(a)

    return (a - mn) / (mx - mn + 1e-12)

def _vectorized_score_for_lottery(X: np.ndarray, lottery_name: str):

    global WORKER_MODELS, WORKER_SCALERS

    scaler = WORKER_SCALERS.get(lottery_name)

    models = WORKER_MODELS.get(lottery_name, {})

    if scaler is not None and X.shape[0] > 0:

        try:

            Xs = scaler.transform(X)

        except Exception:

            Xs = X

    else:

        Xs = X

    n = Xs.shape[0]

    score_if = np.zeros(n, dtype=np.float32)

    score_xgb = np.zeros(n, dtype=np.float32)

    score_lgbm = np.zeros(n, dtype=np.float32)

    score_lstm = np.zeros(n, dtype=np.float32)

    try:

        if models.get("if") is not None:

            score_if = -models["if"].score_samples(Xs).astype(np.float32)

    except Exception:

        score_if = np.zeros(n, dtype=np.float32)

    try:

        if models.get("xgb") is not None:

            score_xgb = models["xgb"].predict_proba(Xs)[:, 1].astype(np.float32)

    except Exception:

        score_xgb = np.zeros(n, dtype=np.float32)

    try:

        if models.get("lgbm") is not None:

            score_lgbm = models["lgbm"].predict_proba(Xs)[:, 1].astype(np.float32)

    except Exception:

        score_lgbm = np.zeros(n, dtype=np.float32)

    try:

        if models.get("lstm") is not None and TENSORFLOW_AVAILABLE:

            X_lstm = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)

            score_lstm = models["lstm"].predict(X_lstm, verbose=0).flatten().astype(np.float32)

    except Exception:

        score_lstm = np.zeros(n, dtype=np.float32)

    s_if = _norm_np(score_if)

    s_xgb = _norm_np(score_xgb)

    s_lgbm = _norm_np(score_lgbm)

    s_lstm = _norm_np(score_lstm)

    # Ensure ML ensemble weights sum to 1.0

    lstm_weight = max(0.0, 1.0 - (ALPHA_IF + BETA_XGB + BETA_LGBM))

    ml_ensemble_raw = ALPHA_IF * s_if + BETA_XGB * s_xgb + BETA_LGBM * s_lgbm + lstm_weight * s_lstm

    raw_norm = _norm_np(ml_ensemble_raw)

    return raw_norm, s_if, s_xgb, s_lgbm, s_lstm

# ---------- Worker batch function ----------

def worker_score_batch_global(batch_args):

    combos, hot_jsons_per_lottery_json, pos_join_per_lottery_json, n_max, k = batch_args

    try:

        hot_map = json.loads(hot_jsons_per_lottery_json)

    except Exception:

        hot_map = {}

    try:

        pos_map = json.loads(pos_join_per_lottery_json)

    except Exception:

        pos_map = {}

    feats_per_lottery = {name: [] for name in WORKER_NAMES}

    combos_out = []

    for c in combos:

        # ensure tuple of ints

        try:

            c_t = tuple(int(x) for x in c)

        except Exception:

            continue

        if not is_plausible_combo(c_t):

            continue

        combos_out.append(c_t)

        for name in WORKER_NAMES:

            hot_json = hot_map.get(name, "{}")

            pos_join = pos_map.get(name, "")

            pos_jsons = tuple(pos_join.split("||")) if isinstance(pos_join, str) and pos_join else ()

            feats_per_lottery[name].append(enhanced_combo_features_tuple(tuple(c_t), hot_json, pos_jsons, n_max))

    if not combos_out:

        return []

    per_lottery_results = {}

    m = len(combos_out)

    for name in WORKER_NAMES:

        feats_list = feats_per_lottery.get(name, [])

        if not feats_list:

            per_lottery_results[name] = {

                "raw": np.zeros(m, dtype=np.float32), "raw_norm": np.zeros(m, dtype=np.float32),

                "hot_scores": np.zeros(m, dtype=np.float32), "entropy": np.zeros(m, dtype=np.float32),

                "parity_score": np.zeros(m, dtype=np.float32), "sum_balance": np.zeros(m, dtype=np.float32),

                "composite": np.zeros(m, dtype=np.float32)

            }

            continue

        X = _features_list_to_array(feats_list)

        if X.shape[0] == 0:

            per_lottery_results[name] = {

                "raw": np.zeros(m, dtype=np.float32), "raw_norm": np.zeros(m, dtype=np.float32),

                "hot_scores": np.zeros(m, dtype=np.float32), "entropy": np.zeros(m, dtype=np.float32),

                "parity_score": np.zeros(m, dtype=np.float32), "sum_balance": np.zeros(m, dtype=np.float32),

                "composite": np.zeros(m, dtype=np.float32)

            }

            continue

        raw_norm, s_if, s_xgb, s_lgbm, s_lstm = _vectorized_score_for_lottery(X, name)

        feats_arr = np.array(feats_list, dtype=np.float32)

        hot_scores = feats_arr[:, -2] if feats_arr.shape[1] >= 12 else np.zeros(len(feats_arr))

        entropy_vals = feats_arr[:, 4] if feats_arr.shape[1] >= 5 else np.zeros(len(feats_arr))

        evens_vals = feats_arr[:, 2] if feats_arr.shape[1] >= 3 else np.zeros(len(feats_arr))

        sums_vals = feats_arr[:, 0] if feats_arr.shape[1] >= 1 else np.zeros(len(feats_arr))

        parity_score = 1.0 - np.abs(evens_vals - (k/2.0)) / (k/2.0 + 1e-9)

        min_sum = sum(range(1, k+1))

        max_sum = sum(range(n_max - k + 1, n_max + 1))

        denom_sum = (max_sum - min_sum) or 1.0

        sum_balance_score = 1.0 - np.abs(sums_vals - ((min_sum + max_sum)/2.0)) / denom_sum

        # Composite: ML dominates (weight 1.0) and stats refine it

        composite = (1.0 * raw_norm +

                     GAMMA_HOT * _norm_np(hot_scores) +

                     DELTA_ENT * _norm_np(entropy_vals) +

                     EPS_PAR * parity_score +

                     ZETA_SUM * sum_balance_score)

        per_lottery_results[name] = {

            "raw": raw_norm, "raw_norm": raw_norm,

            "hot_scores": hot_scores, "entropy": entropy_vals,

            "parity_score": parity_score, "sum_balance": sum_balance_score,

            "composite": composite

        }

    composites_stack = []

    for name in WORKER_NAMES:

        comps = per_lottery_results.get(name, {}).get("composite", np.zeros(m, dtype=np.float32))

        if comps.shape[0] != m:

            comps = np.resize(comps, m)

        composites_stack.append(comps)

    composites_stack = np.stack(composites_stack, axis=1)

    global_composite = np.mean(composites_stack, axis=1)

    results = []

    for idx, c in enumerate(combos_out):

        per_lottery_detail = {}

        for name in WORKER_NAMES:

            detail = per_lottery_results[name]

            per_lottery_detail[name] = {

                "composite": float(detail["composite"][idx]) if detail["composite"].size > idx else 0.0,

                "raw": float(detail["raw"][idx]) if detail["raw"].size > idx else 0.0,

                "hot_score": float(detail["hot_scores"][idx]) if detail["hot_scores"].size > idx else 0.0,

                "entropy": float(detail["entropy"][idx]) if detail["entropy"].size > idx else 0.0,

            }

        results.append({

            "combo": list(c),

            "suma": int(sum(c)),

            "global_composite": float(global_composite[idx]),

            "per_lottery": per_lottery_detail

        })

    try:

        gc.collect()

    except Exception:

        pass

    return results

# ---------- Candidate generation ----------

def generate_candidates_importance_combined(n_max: int, k: int, n_samples: int, hot_per_lottery: Dict[str, Dict[str, float]], rng_seed: int = SEED):

    rng = np.random.default_rng(rng_seed)

    combined = np.zeros(n_max, dtype=float)

    for name, hot in hot_per_lottery.items():

        for num_s, w in hot.items():

            try:

                i = int(num_s) - 1

                if 0 <= i < n_max:

                    combined[i] += float(w)

            except Exception:

                continue

    base_probs = np.ones(n_max, dtype=float) + combined * 8.0

    base_probs = base_probs / base_probs.sum()

    number_range = np.arange(1, n_max+1)

    candidates = set()

    attempts = 0

    max_attempts = max(10000, n_samples * 8)

    batch_size = 4096

    # make safe (avoid zero probs)

    base_probs_safe = base_probs.copy()

    base_probs_safe[base_probs_safe == 0] = 1e-12

    base_probs_safe = base_probs_safe / base_probs_safe.sum()

    while len(candidates) < n_samples and attempts < max_attempts:

        try:

            for _ in range(batch_size):

                sel = rng.choice(number_range, size=k, replace=False, p=base_probs_safe)

                c = tuple(sorted(int(x) for x in sel))

                if is_plausible_combo(c):

                    candidates.add(c)

                if len(candidates) >= n_samples:

                    break

            attempts += batch_size

        except Exception:

            idx_matrix = rng.choice(number_range, size=(batch_size, k), p=base_probs_safe, replace=True)

            for row in idx_matrix:

                c = tuple(sorted(set(int(x) for x in row)))

                if len(c) == k and is_plausible_combo(c):

                    candidates.add(c)

                if len(candidates) >= n_samples:

                    break

            attempts += batch_size

    if not candidates:

        raise RuntimeError("No candidates generated via importance sampling")

    result = list(candidates)

    if len(result) > n_samples:

        rng.shuffle(result)

        result = result[:n_samples]

    return result

def expand_neighbors(top_combos: List[Tuple[int, ...]], n_max: int, k: int, per_combo: int = NEIGHBORS_PER_COMBO, rng_seed: int = SEED):

    rng = np.random.default_rng(rng_seed)

    neighbors = set()

    # limit number of top combos to expand to avoid explosion

    limit_top = min(len(top_combos), max(500, PRERANK_TOP // 20))

    for combo in top_combos[:limit_top]:

        base = list(combo)

        for _ in range(per_combo):

            c = base.copy()

            # randomly replace 1-2 positions guided by hot probabilities of the combo's lotteries unknown here

            num_changes = rng.choice([1,1,2], p=[0.45,0.45,0.10])

            for _ in range(num_changes):

                idx = rng.integers(0, k)

                replacement = int(rng.integers(1, n_max+1))

                c[idx] = replacement

            c_sorted = tuple(sorted(set(int(x) for x in c)))

            if len(c_sorted) == k and is_plausible_combo(c_sorted):

                neighbors.add(c_sorted)

            if len(neighbors) >= TARGET_EVAL:

                break

        if len(neighbors) >= TARGET_EVAL:

            break

    if not neighbors:

        raise RuntimeError("No neighbors generated in expansion")

    result = list(neighbors)

    if len(result) > TARGET_EVAL:

        rng.shuffle(result)

        result = result[:TARGET_EVAL]

    return result

# ---------- Progress logger ----------

def _log_progress_eta(phase: str, start_time: float, processed: int, total: int, last_pct: int, prefix: str = "") -> int:

    if total == 0:

        return last_pct

    pct = int(processed / total * 100)

    if pct >= last_pct + 5 or processed == total:

        elapsed = time.time() - start_time

        if processed > 0:

            eta = (elapsed / processed) * (total - processed)

            eta_ts = datetime.now() + timedelta(seconds=eta)

            logger.info(f"{prefix} {phase} progress: {pct}% ({processed}/{total}) - Elapsed: {int(elapsed)}s - ETA: {eta_ts.strftime('%Y-%m-%d %H:%M:%S')}")

        else:

            logger.info(f"{prefix} {phase} progress: {pct}% ({processed}/{total}) - Elapsed: {int(elapsed)}s")

        last_pct = pct

    return last_pct

# ---------- Orchestration pipeline ----------

def sample_prerank_and_expand(all_histories: Dict[str, pd.DataFrame], model_files: Dict[str, str], n_prerank: int = PRERANK_SAMPLES):

    start_time = time.time()

    logger.info("Fase 1: Preranking - Generando candidatos por importancia")

    hot_map = {}

    pos_map = {}

    for name, df in all_histories.items():

        hot_map[name] = build_hot_freq(df, k=LOTTERIES[name]["k"])

        pos_map[name] = build_pos_freq(df, k=LOTTERIES[name]["k"])

    n_max = next(iter(LOTTERIES.values()))["n_max"]

    k = next(iter(LOTTERIES.values()))["k"]

    # Generate initial candidates

    candidates = generate_candidates_importance_combined(n_max, k, n_prerank, hot_map, rng_seed=SEED)

    logger.info(f"Prerank: {len(candidates)} candidatos generados")

    # Evaluate prerank with workers (reuse worker_score_batch_global)

    hot_json_map = {name: json.dumps(hot_map[name]) for name in hot_map}

    pos_join_map = {name: "||".join(json.dumps(p) for p in pos_map[name]) for name in pos_map}

    chunk_size = max(1, int(math.ceil(len(candidates) / WORKERS)))

    batches = []

    for i in range(0, len(candidates), chunk_size):

        batches.append((candidates[i:i+chunk_size], json.dumps(hot_json_map), json.dumps(pos_join_map), n_max, k))

    results = []

    ctx = get_context("spawn")

    init_args = (json.dumps(model_files), json.dumps(list(LOTTERIES.keys())))

    last_pct = -5

    processed = 0

    total = len(batches)

    try:

        with ctx.Pool(processes=WORKERS, initializer=worker_initializer, initargs=init_args) as pool:

            for res in pool.imap_unordered(worker_score_batch_global, batches):

                processed += 1

                if res:

                    results.extend(res)

                if processed % max(1, total//20) == 0:

                    gc.collect()

                last_pct = _log_progress_eta("PRERANK", start_time, processed, total, last_pct, prefix="Jefe Maestro (prerank)")

    except Exception as e:

        logger.error(f"Prerank pool error: {e}. Fallback single-threaded.")

        worker_initializer(json.dumps(model_files), json.dumps(list(LOTTERIES.keys())))

        for b in batches:

            res = worker_score_batch_global(b)

            if res:

                results.extend(res)

    if not results:

        raise RuntimeError("No results produced during prerank")

    df = pd.DataFrame(results)

    df_preranked = df.sort_values("global_composite", ascending=False).drop_duplicates(subset=["combo"]).reset_index(drop=True)

    top_prerank = df_preranked.head(PRERANK_TOP)

    logger.info(f"Prerank: top {len(top_prerank)} retenidas para expansión")

    # Expansion

    logger.info("Fase 2: Expansión de vecindarios alrededor de top prerank")

    top_combos = [tuple(int(x) for x in row["combo"]) for _, row in top_prerank.iterrows()]

    neighbors = expand_neighbors(top_combos, n_max, k, per_combo=NEIGHBORS_PER_COMBO, rng_seed=SEED)

    logger.info(f"Expansión: generados {len(neighbors)} vecinos")

    # Combine unique set for final evaluation (top_prerank combos + neighbors)

    final_candidates_set = set(tuple(int(x) for x in row["combo"]) for _, row in top_prerank.iterrows())

    final_candidates_set.update(neighbors)

    final_candidates = list(final_candidates_set)

    logger.info(f"Fase 3: Evaluación final - {len(final_candidates)} candidatos a evaluar (objetivo <= {TARGET_EVAL})")

    return final_candidates, hot_map, pos_map

def final_evaluate_and_select(final_candidates: List[Tuple[int,...]], hot_map: Dict[str, Dict[str, float]], pos_map: Dict[str, List[Dict[str,float]]], model_files: Dict[str,str], top_k: int = TOP_K):

    n_max = next(iter(LOTTERIES.values()))["n_max"]

    k = next(iter(LOTTERIES.values()))["k"]

    # Prepare batches

    chunk_size = max(1, int(math.ceil(len(final_candidates) / WORKERS)))

    batches = []

    hot_json_map = {name: json.dumps(hot_map[name]) for name in hot_map}

    pos_join_map = {name: "||".join(json.dumps(p) for p in pos_map[name]) for name in pos_map}

    for i in range(0, len(final_candidates), chunk_size):

        batches.append((final_candidates[i:i+chunk_size], json.dumps(hot_json_map), json.dumps(pos_join_map), n_max, k))

    results = []

    start_time = time.time()

    last_pct = -5

    processed = 0

    total = len(batches)

    ctx = get_context("spawn")

    init_args = (json.dumps(model_files), json.dumps(list(LOTTERIES.keys())))

    try:

        with ctx.Pool(processes=WORKERS, initializer=worker_initializer, initargs=init_args) as pool:

            for res in pool.imap_unordered(worker_score_batch_global, batches):

                processed += 1

                if res:

                    results.extend(res)

                if processed % max(1, total//20) == 0:

                    gc.collect()

                last_pct = _log_progress_eta("FINAL_EVAL", start_time, processed, total, last_pct, prefix="Jefe Maestro (final eval)")

    except Exception as e:

        logger.error(f"Final eval pool error: {e}. Fallback single-threaded.")

        worker_initializer(json.dumps(model_files), json.dumps(list(LOTTERIES.keys())))

        for b in batches:

            res = worker_score_batch_global(b)

            if res:

                results.extend(res)

    if not results:

        raise RuntimeError("No results in final evaluation")

    df = pd.DataFrame(results)

    df_sorted = df.sort_values("global_composite", ascending=False).drop_duplicates(subset=["combo"]).head(top_k).reset_index(drop=True)

    stats = {"count": len(df), "mean_score": float(df["global_composite"].mean()), "std_score": float(df["global_composite"].std()), "execution_time": time.time() - start_time}

    return df_sorted, stats

# ---------- Persistence & Email ----------

def save_predictions_global(prediction_date: str, aggregated: pd.DataFrame):

    history = load_predictions_history()

    key = "GlobalUnified"

    history_entry = {"date": prediction_date, "predicted": []}

    for _, row in aggregated.iterrows():

        history_entry["predicted"].append({

            "combo": [int(x) for x in row["combo"]],

            "global_composite": float(row["global_composite"]),

            "per_lottery": row.get("per_lottery", {})

        })

    if key not in history:

        history[key] = []

    history[key].append(history_entry)

    history[key] = history[key][-20:]

    try:

        with open(PREDICTIONS_HISTORY_FILE, 'w', encoding='utf-8') as f:

            json.dump(history, f, default=safe_json_convert, ensure_ascii=False, indent=2)

        logger.info(f"Predictions history saved to {PREDICTIONS_HISTORY_FILE}")

    except Exception as e:

        logger.error(f"Error saving predictions history: {e}")

        send_telegram_alert(f"Error saving history: {e}")

def df_to_html_table_global(df: pd.DataFrame, title: str, limit: int = 20):

    lines = []

    lines.append(f"<h3>{title}</h3>")

    lines.append('<table border="1" cellpadding="5" cellspacing="0">')

    lines.append("<thead><tr><th>#</th><th>Combinación</th><th>Global Composite</th><th>Suma</th><th>Per Lottery Details</th></tr></thead><tbody>")

    for i, row in df.head(limit).iterrows():

        combo = " ".join(f"{int(x):02d}" for x in row["combo"])

        composite = float(row.get("global_composite", 0.0))

        suma = int(row.get("suma", sum(row["combo"]) if isinstance(row["combo"], (list,tuple)) else 0))

        per = json.dumps(row.get("per_lottery", {}), ensure_ascii=False)

        lines.append(f"<tr><td>{i+1}</td><td><b>{combo}</b></td><td>{composite:.6f}</td><td>{suma}</td><td>{per}</td></tr>")

    lines.append("</tbody></table>")

    return "\n".join(lines)

def send_email_gmail_unified_global(aggregated: pd.DataFrame, system_stats: Dict, ts: str):

    if not EMAIL_FROM or not EMAIL_PASS or not EMAIL_TO:

        logger.warning("Email credentials not set - skipping prediction email")

        return False

    subject = f"Jefe Maestro - Top combinaciones unificadas (Elite v6.0) - {ts}"

    html_lines = []

    html_lines.append(f"<h2>Jefe Maestro - Top combinaciones unificadas (Elite v6.0)</h2>")

    html_lines.append(f"<p>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>")

    html_lines.append(df_to_html_table_global(aggregated, "Top combinaciones - Global Composite (una combinación para los 3 sorteos)"))

    html_lines.append("<h3>System stats</h3>")

    html_lines.append(f"<pre>{json.dumps(system_stats, indent=2)}</pre>")

    html_body = "\n".join(html_lines)

    msg = MIMEMultipart("alternative")

    msg["Subject"] = subject

    msg["From"] = EMAIL_FROM

    msg["To"] = EMAIL_TO

    part1 = MIMEText(html_body, "html")

    msg.attach(part1)

    try:

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:

            server.ehlo()

            server.starttls()

            server.ehlo()

            server.login(EMAIL_FROM, EMAIL_PASS)

            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())

        logger.info(f"Unified global email sent to {EMAIL_TO}")

        return True

    except Exception as e:

        logger.error(f"Error sending unified global email: {e}")

        send_telegram_alert(f"Error sending unified global email: {e}")

        return False

# ---------- Main orchestration ----------

def main():

    parser = argparse.ArgumentParser(description="Jefe Maestro Elite v6.0 - Global Unified Predictor")

    parser.add_argument("--fast-mode", action="store_true", help="Usar modo rápido (menor número de candidatos)")

    args = parser.parse_args()

    logger.info("✅ Iniciando Jefe Maestro Elite v6.0...")

    system_info = {"version": "v6.0_elite_predictor", "models": ["IsolationForest", "XGBoost", "LightGBM", "LSTM" if TENSORFLOW_AVAILABLE else None], "timestamp": datetime.now().isoformat(), "workers": WORKERS, "lotteries": list(LOTTERIES.keys())}

    system_info["models"] = [m for m in system_info["models"] if m is not None]

    # 1) Load histories

    try:

        all_histories = load_all_histories_strict()

    except SystemExit:

        return

    except Exception as e:

        abort_no_data(f"Error cargando historiales: {e}")

    # 2) Load or train models

    model_files = {}

    models_map = {}

    for name, df in all_histories.items():

        try:

            models = load_or_train_model(df, name)

            models_map[name] = models

            model_files[name] = MODEL_FILE_TEMPLATE.format(name=name)

        except Exception as e:

            abort_no_data(f"No se pudo cargar/entrenar modelo para {name}: {e}")

    # 3) Prerank & expansion

    try:

        n_prerank = int(PRERANK_SAMPLES/4) if args.fast_mode else PRERANK_SAMPLES

        final_candidates, hot_map, pos_map = sample_prerank_and_expand(all_histories, model_files, n_prerank=n_prerank)

    except Exception as e:

        abort_no_data(f"Pipeline prerank/expand falló: {e}")

    # 4) Final evaluation

try:

    df_global_top, stats = final_evaluate_and_select(
        final_candidates,
        hot_map,
        pos_map,
        model_files,
        top_k=TOP_K
    )

    # ===============================
    # MULTI CLUSTER SELECTION LAYER
    # ===============================

    try:
        df_global_top = apply_multi_cluster_layer(df_global_top)
        logger.info("Multi-cluster expansion aplicado correctamente")
    except Exception as e:
        logger.warning(f"Multi-cluster layer falló: {e}")

except Exception as e:

        abort_no_data(f"Evaluación final falló: {e}")

        # 5) Save aggregated

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    aggregated_fn = os.path.join(
        RESULTS_DIR,
        f"aggregated_global_v6_{ts}.json"
    )
    try:

        aggregated_to_save = []

        for _, row in df_global_top.iterrows():

            aggregated_to_save.append({

                "combo_str": " ".join(f"{int(x):02d}" for x in row["combo"]),

                "combo": [int(x) for x in row["combo"]],

                "global_composite": float(row["global_composite"]),

                "suma": int(row["suma"]),

                "per_lottery": row.get("per_lottery", {})

            })

        with open(aggregated_fn, "w", encoding="utf-8") as f:

            json.dump({"aggregated": aggregated_to_save, "system_info": system_info, "stats": stats}, f, default=safe_json_convert, ensure_ascii=False, indent=2)

        logger.info(f"Aggregated saved: {aggregated_fn}")

    except Exception as e:

        logger.error(f"Error saving aggregated: {e}")

    # 6) Save predictions history

    save_predictions_global(datetime.now().strftime('%Y-%m-%d'), df_global_top)

    # 7) Send email

    try:

        send_email_gmail_unified_global(df_global_top, {"system_info": system_info, "stats": stats}, ts)

    except Exception as e:

        logger.error(f"Error sending final email: {e}")

        send_telegram_alert(f"Error sending final email: {e}")

    logger.info("Proceso completado. Revisa results/ y tu correo.")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd


# ============================================================
# AUTO DISCOVERY OF CLUSTERS
# ============================================================

def find_optimal_clusters(features, max_k=20):

    best_k = 8
    best_score = -1

    for k in range(6, max_k):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(features)

        score = silhouette_score(features, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


# ============================================================
# DISCOVER STRUCTURAL CLUSTERS
# ============================================================

def discover_structural_clusters(df):

    features = build_cluster_feature_matrix(df)

    scaler = StandardScaler()

    features_scaled = scaler.fit_transform(features)

    optimal_k = find_optimal_clusters(features_scaled)

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=30
    )

    labels = kmeans.fit_predict(features_scaled)

    df = df.copy()

    df["cluster_id"] = labels

    return df, kmeans


# ============================================================
# CLUSTER EXPANSION (CORE IMPROVEMENT)
# ============================================================

def expand_clusters(df_clustered, ranked_clusters):

    results = []

    used = set()

    for cid in ranked_clusters["cluster_id"]:

        cluster_df = df_clustered[
            df_clustered["cluster_id"] == cid
        ].sort_values(
            "score_norm",
            ascending=False
        )

        top_rows = cluster_df.head(TOP_PER_CLUSTER * 2)

        for _, row in top_rows.iterrows():

            combo = tuple(row["combo"])

            if combo in used:
                continue

            results.append(row)

            used.add(combo)

            if len(results) >= TOP_CLUSTERS * TOP_PER_CLUSTER:
                break

    return pd.DataFrame(results)

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        logger.critical(f"Excepción fatal en main: {e}", exc_info=True)

        send_telegram_alert(f"Excepción fatal en Jefe Maestro v6.0: {e}")

        sys.exit(1)
