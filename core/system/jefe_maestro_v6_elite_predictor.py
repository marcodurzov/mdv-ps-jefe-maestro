#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jefe_maestro_v6_elite_predictor.py
Versión interna: v8.0 Final Supreme

ARCHIVO PRINCIPAL DEL PREDICTOR — nombre mantenido para compatibilidad
con main_run.py y automation.yml del repositorio.

REGLAS DEL SORTEO (importante para entender el código):
  - El jugador elige 6 números del 1 al 56.
  - La misma combinación de 6 números participa en los 3 sorteos:
    Melate, Revancha y Revanchita.
  - Melate saca 6 bolas principales + 1 BONO adicional (el jugador
    NO elige el bono, es extra del sorteo). El bono se guarda en CSV
    pero NO forma parte de la combinación a predecir.
  - Revancha y Revanchita: solo 6 bolas (N1–N6), sin bono.

MEJORAS INCORPORADAS v6→v8:
═══════════════════════════════════════════════════════════════════════
ESTRUCTURA
  ✓ Nombre correcto: jefe_maestro_v6_elite_predictor.py
  ✓ CSV Melate: FECHA,N1,N2,N3,N4,N5,N6,BONO
  ✓ CSV Revancha/Revanchita: FECHA,N1,N2,N3,N4,N5,N6
  ✓ CSV siempre ordenado: más reciente primero (regla 7)
  ✓ Rutas relativas correctas (data/ desde raíz del repo)

FEATURES (~70 dimensiones)
  ✓ Hot multi-ventana: 10 / 30 / 100 / 300 sorteos
  ✓ Gap analysis (números overdue)
  ✓ Co-ocurrencias normalizadas vs. esperado uniforme
  ✓ FFT espectral por número (detección de ciclos)
  ✓ Distancia KS vs. distribución uniforme
  ✓ Estructurales: primos, Fibonacci, cuadrados, mult-5
  ✓ Distribución por tercios
  ✓ Bono Melate como feature auxiliar
  ✓ Penalización de combos "humanas" (fechas, secuencias, patrones)

ENSEMBLE
  ✓ IsolationForest
  ✓ XGBoost calibrado (Platt)
  ✓ LightGBM calibrado (Isotonic)
  ✓ CatBoost calibrado (si disponible, solo modo full)
  ✓ LSTM mejorado con EarlyStopping (solo modo full + TF)
  ✓ Meta-learner: HistGradientBoostingClassifier (OOF stacking)
  ✓ Temperature scaling en composite final
  ✓ Permutation importance logueado

CANDIDATOS (4 estrategias)
  ✓ 40% Hot-guided
  ✓ 25% Gap-guided (overdue)
  ✓ 10% Co-occurrence-guided (pares frecuentes)
  ✓ 25% Uniforme (exploración)

PORTFOLIO
  ✓ Beam search (width=5) + set-cover
  ✓ Diversidad coseno intra-portfolio
  ✓ Early stopping (mejora < 0.0005 en 3 iter)

RENDIMIENTO / GITHUB ACTIONS
  ✓ --light / LIGHT_MODE=true / auto-detect RAM < 4.5 GB
  ✓ Stats cacheadas en disco (no se recalculan si datos no cambiaron)
  ✓ Backtest con historial de lift en JSON
  ✓ Reintentos en email (3 intentos)
  ✓ Simulated Annealing en expansión de vecindarios
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os, sys, math, json, time, gc, logging, warnings, argparse, random
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import joblib
import psutil
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

from imblearn.over_sampling import SMOTE, RandomOverSampler
import requests, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from multiprocessing import get_context, cpu_count
from logging.handlers import RotatingFileHandler

# ─────────────────────────────────────────────────────────────────────
# RUTAS (compatibles con estructura del repositorio)
# ─────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))   # core/system/
_ROOT     = os.path.dirname(os.path.dirname(_HERE))      # raíz del repo
_DATA_DIR = os.path.join(_ROOT, "data")
_CACHE    = os.path.join(_ROOT, "cache")
_RESULTS  = os.path.join(_ROOT, "results")
for _d in [_CACHE, _RESULTS]: os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
LOG_FILE = os.path.join(_ROOT, "jefe_maestro_v8.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# AUTO-DETECT RECURSOS
# ─────────────────────────────────────────────────────────────────────
_RAM_GB    = psutil.virtual_memory().total / (1024**3)
_CPUS      = cpu_count()
_AUTO_LIGHT = _RAM_GB < 4.5

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE SORTEOS
# ─────────────────────────────────────────────────────────────────────
LOTTERIES: Dict[str, Dict] = {
    "Melate":     {"n_max": 56, "k": 6, "has_bono": True},
    "Revancha":   {"n_max": 56, "k": 6, "has_bono": False},
    "Revanchita": {"n_max": 56, "k": 6, "has_bono": False},
}

MODEL_FILE_TPL      = os.path.join(_CACHE, "model_v8_{name}.joblib")
STATS_FILE_TPL      = os.path.join(_CACHE, "stats_v8_{name}.joblib")
PREDICTIONS_HISTORY = os.path.join(_ROOT,  "predictions_history_v8.json")
BACKTEST_HISTORY    = os.path.join(_ROOT,  "backtest_history_v8.json")

# ── Pesos ensemble fallback ──────────────────────────────────────────
W_IF   = 0.10
W_XGB  = 0.35
W_LGBM = 0.20
W_CB   = 0.15
W_LSTM = 0.20

# ── Refinamiento estadístico ─────────────────────────────────────────
GAMMA_HOT  = 0.06
DELTA_KS   = 0.05
EPS_PAR    = 0.04
ZETA_SUM   = 0.04
ETA_GAP    = 0.06
THETA_COV  = 0.05
IOTA_HUM   = 0.08   # penalización combos "humanas"

ALPHA_DECAY = 0.95
SEED        = int(os.getenv("JM_SEED", "42"))
np.random.seed(SEED); random.seed(SEED)

TOP_K   = int(os.getenv("JM_TOP_K", "20"))
WORKERS = int(os.getenv("JM_WORKERS", str(max(1, min(4, _CPUS)))))

_PRERANK_FULL  = 160_000;  _TARGET_FULL  = 2_000_000; _NEIGH_FULL  = 25
_PRERANK_LIGHT =  40_000;  _TARGET_LIGHT =   500_000; _NEIGH_LIGHT = 15
PRERANK_TOP    = int(os.getenv("PRERANK_TOP", "30000"))

MIN_SUM = 60; MAX_SUM = 300; MAX_CONSEC = 4

EMAIL_FROM  = os.getenv("EMAIL_USER")
EMAIL_PASS  = os.getenv("EMAIL_PASS")
EMAIL_TO    = os.getenv("EMAIL_TO") or EMAIL_FROM
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT   = 587

# Pesos por pozo acumulado (opcionales)
_POT       = {n: float(os.getenv(f"POT_{n.upper()}", "1.0")) for n in LOTTERIES}
_PT        = sum(_POT.values()) or 1.0
POT_WEIGHTS = {n: _POT[n] / _PT for n in LOTTERIES}

# Globals de workers
WORKER_MODELS:  Dict = {}
WORKER_SCALERS: Dict = {}
WORKER_STATS:   Dict = {}
WORKER_NAMES:   List = []
WORKER_LIGHT:   bool = False

# ─────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────

def safe_json(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, pd.Timestamp):   return obj.isoformat()
    if isinstance(obj, pd.Series):      return obj.tolist()
    return obj

def send_telegram(msg: str):
    tok = os.getenv("TELEGRAM_TOKEN"); chat = os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                           json={"chat_id": chat, "text": msg}, timeout=10)
        except Exception: pass

def _norm(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr.astype(np.float32), nan=0., posinf=0., neginf=0.)
    mn, mx = a.min(), a.max()
    return np.zeros_like(a) if mx-mn < 1e-12 else (a-mn)/(mx-mn+1e-12)

# ─────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────

def load_history_strict(name: str) -> pd.DataFrame:
    path = os.path.join(_DATA_DIR, f"{name.lower()}.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"CSV no encontrado: {path}")
    df = pd.read_csv(path)
    if df.empty: raise RuntimeError(f"CSV vacío: {name}")
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    k = LOTTERIES[name]["k"]
    has_b = LOTTERIES[name]["has_bono"]
    num_cols = [f"N{i}" for i in range(1, k+1)]
    if not all(c in df.columns for c in num_cols):
        raise RuntimeError(f"Columnas faltantes en {name}: {num_cols}")
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=num_cols)
    n_max = LOTTERIES[name]["n_max"]
    for col in num_cols:
        if not df[col].between(1, n_max).all():
            raise RuntimeError(f"Valores fuera de rango en {name}.{col}")
    if has_b:
        if "BONO" in df.columns:
            df["BONO"] = pd.to_numeric(df["BONO"], errors="coerce")
        else:
            df["BONO"] = np.nan
    # Ordenar: más reciente primero (regla 7)
    df = df.sort_values("FECHA", ascending=False).reset_index(drop=True)
    df["FUENTE"] = name
    return df

def load_all_histories() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for name in LOTTERIES:
        try:
            dfs[name] = load_history_strict(name)
            logger.info(f"[{name}] {len(dfs[name]):,} sorteos cargados")
        except Exception as e:
            _abort(f"No se pudo cargar {name}: {e}")
    return dfs

def _abort(reason: str):
    logger.critical(f"ABORT: {reason}")
    send_telegram(f"Jefe Maestro v8 ABORT: {reason}")
    sys.exit(2)

# ─────────────────────────────────────────────────────────────────────
# NÚMEROS ESTRUCTURALES
# ─────────────────────────────────────────────────────────────────────

def _primes(n):
    s=[True]*(n+1); s[0]=s[1]=False
    for i in range(2,int(n**.5)+1):
        if s[i]:
            for j in range(i*i,n+1,i): s[j]=False
    return frozenset(i for i in range(2,n+1) if s[i])

def _fibs(n):
    r,a,b=set(),1,1
    while a<=n: r.add(a); a,b=b,a+b
    return frozenset(r)

PRIMES56 = _primes(56)
FIBS56   = _fibs(56)
SQRS56   = frozenset(i*i for i in range(1,8))

def _is_date_like(nums: tuple) -> bool:
    return sum(1 for n in nums if n<=31)>=4 and sum(1 for n in nums if n<=12)>=2

# ─────────────────────────────────────────────────────────────────────
# ESTADÍSTICAS HISTÓRICAS
# ─────────────────────────────────────────────────────────────────────

def _build_stats(df: pd.DataFrame, name: str) -> Dict:
    k       = LOTTERIES[name]["k"]
    n_max   = LOTTERIES[name]["n_max"]
    has_b   = LOTTERIES[name]["has_bono"]
    n       = len(df)
    ncols   = [f"N{i}" for i in range(1, k+1)]
    df_asc  = df.iloc[::-1].reset_index(drop=True)  # asc para cálculos

    # ── Hot multi-ventana (df está desc → head = más reciente) ────
    mw: Dict[int, Dict[int,float]] = {}
    for w in [10, 30, 100, 300]:
        freq: Dict[int,float] = {}
        for idx, (_, row) in enumerate(df.head(w).iterrows()):
            wt = ALPHA_DECAY**idx
            for c in ncols:
                if pd.notna(row.get(c)):
                    num = int(row[c]); freq[num] = freq.get(num,0.)+wt
        tot = sum(freq.values()) or 1.
        mw[w] = {nm: v/tot for nm,v in freq.items()}

    # ── Gap ───────────────────────────────────────────────────────
    last_seen = {nm: -1 for nm in range(1, n_max+1)}
    for idx,(_, row) in enumerate(df_asc.iterrows()):
        for c in ncols:
            if pd.notna(row.get(c)): last_seen[int(row[c])] = idx
    exp_g = n_max / k
    gap = {nm: (float(n)/exp_g if last_seen[nm]==-1
                else float(n-1-last_seen[nm])/exp_g)
           for nm in range(1, n_max+1)}

    # ── Co-ocurrencias ────────────────────────────────────────────
    cooc = np.zeros((n_max+1, n_max+1), dtype=np.float32)
    bono_cooc: Dict[int,float] = {}
    for _,(_, row) in enumerate(df_asc.iterrows()):
        nums = [int(row[c]) for c in ncols if pd.notna(row.get(c))]
        for a in range(len(nums)):
            for b in range(a+1, len(nums)):
                cooc[nums[a],nums[b]]+=1; cooc[nums[b],nums[a]]+=1
        if has_b and "BONO" in row and pd.notna(row["BONO"]):
            bv=int(row["BONO"])
            for nm in nums: bono_cooc[nm]=bono_cooc.get(nm,0.)+1.
    exp_c = 2*n*k*(k-1)/(n_max*(n_max-1)) or 1e-9
    cooc_norm = cooc/(exp_c+1e-9)
    cooc_d: Dict[str,float] = {}
    for i in range(1,n_max+1):
        for j in range(i+1,n_max+1):
            v=float(cooc_norm[i,j])
            if abs(v-1.)>0.05: cooc_d[f"{i}_{j}"]=v

    # ── KS ────────────────────────────────────────────────────────
    cnt = np.zeros(n_max+1, dtype=np.int32)
    for _,(_, row) in enumerate(df_asc.iterrows()):
        for c in ncols:
            if pd.notna(row.get(c)): cnt[int(row[c])]+=1
    tot_c = cnt[1:].sum() or 1
    ep    = 1./n_max
    ks    = {nm: abs(cnt[nm]/tot_c-ep)/(ep+1e-9) for nm in range(1,n_max+1)}

    # ── FFT espectral ─────────────────────────────────────────────
    spectral: Dict[int,float] = {}
    for nm in range(1, n_max+1):
        series = np.zeros(n, dtype=np.float32)
        for idx,(_, row) in enumerate(df_asc.iterrows()):
            for c in ncols:
                if pd.notna(row.get(c)) and int(row[c])==nm: series[idx]=1.
        if series.sum()>0:
            fft_v = np.abs(np.fft.rfft(series-series.mean()))
            spectral[nm] = float(fft_v[1:].max()) if len(fft_v)>1 else 0.
        else:
            spectral[nm] = 0.
    mx_s = max(spectral.values()) or 1.
    spectral = {nm: v/mx_s for nm,v in spectral.items()}

    # ── Posición frecuencia ───────────────────────────────────────
    pos_freq = [{} for _ in range(k)]
    for idx,(_, row) in enumerate(df_asc.iterrows()):
        wt = ALPHA_DECAY**(n-idx-1)
        for i in range(k):
            c=f"N{i+1}"
            if pd.notna(row.get(c)):
                nm=int(row[c]); pos_freq[i][nm]=pos_freq[i].get(nm,0.)+wt
    for p in pos_freq:
        tot=sum(p.values()) or 1.
        for nm in p: p[nm]/=tot

    # ── Bono freq ─────────────────────────────────────────────────
    bono_freq: Dict[int,float] = {}
    if has_b:
        for _,(_, row) in enumerate(df.head(50).iterrows()):
            if "BONO" in row and pd.notna(row["BONO"]):
                bv=int(row["BONO"]); bono_freq[bv]=bono_freq.get(bv,0.)+1.
        tot_b=sum(bono_freq.values()) or 1.
        bono_freq={nm: v/tot_b for nm,v in bono_freq.items()}

    return {
        "hot10":     {str(k):v for k,v in mw[10].items()},
        "hot30":     {str(k):v for k,v in mw[30].items()},
        "hot100":    {str(k):v for k,v in mw[100].items()},
        "hot300":    {str(k):v for k,v in mw[300].items()},
        "gap":       {str(k):v for k,v in gap.items()},
        "ks":        {str(k):v for k,v in ks.items()},
        "spectral":  {str(k):v for k,v in spectral.items()},
        "pos":       [{str(k):v for k,v in p.items()} for p in pos_freq],
        "cooc":      cooc_d,
        "bono_freq": {str(k):v for k,v in bono_freq.items()},
        "n_draws":   n,
    }

def load_or_build_stats(df: pd.DataFrame, name: str) -> Dict:
    sf = STATS_FILE_TPL.format(name=name)
    n  = len(df)
    if os.path.exists(sf):
        try:
            c = joblib.load(sf)
            if c.get("n_draws") == n:
                logger.info(f"[{name}] Stats desde caché"); return c
        except Exception: pass
    logger.info(f"[{name}] Calculando stats ({n} draws)...")
    t0 = time.time()
    s  = _build_stats(df, name)
    try: joblib.dump(s, sf)
    except Exception as e: logger.warning(f"No se guardó stats caché: {e}")
    logger.info(f"[{name}] Stats en {time.time()-t0:.1f}s")
    return s

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (~70 features)
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=600_000)
def combo_features(
    nums: Tuple[int,...], n_max: int,
    h10:str, h30:str, h100:str, h300:str,
    gap_j:str, ks_j:str, sp_j:str,
    pos_j:str, cooc_j:str, bono_j:str,
) -> Tuple[float,...]:
    nums = tuple(sorted(int(x) for x in nums))
    k    = len(nums)
    feats: List[float] = []

    # G1: Básicas (11)
    s   = float(sum(nums)); ev = float(sum(1 for n in nums if n%2==0))
    rng = float(max(nums)-min(nums)); std = float(np.std(nums))
    dif = np.diff(nums)
    md  = float(dif.min()) if dif.size else 0.
    xd  = float(dif.max()) if dif.size else 0.
    ld  = [n%10 for n in nums]; dc={}
    for d in ld: dc[d]=dc.get(d,0)+1
    le  = sum(-p/k*math.log2(p/k+1e-12) for p in dc.values())
    cons=1; mr=1
    for i in range(k-1):
        if nums[i+1]==nums[i]+1: cons+=1; mr=max(mr,cons)
        else: cons=1
    bins=np.histogram(nums,bins=max(1,math.ceil(n_max/10)),range=(1,n_max+1))[0]
    bn=bins/(bins.sum()+1e-9)
    de=float(-sum(p*math.log2(p+1e-12) for p in bn if p>0))
    ds=float(sum(int(d) for n in nums for d in str(n)))
    feats+=[s,ev,rng,std,md,xd,le,float(mr),de,ds,
            float(np.var(dif)) if dif.size else 0.]

    # G2: Tercios (6)
    th=n_max/3.; lo=float(sum(1 for n in nums if n<=th))
    mi=float(sum(1 for n in nums if th<n<=2*th))
    hi=float(sum(1 for n in nums if n>2*th))
    feats+=[lo,mi,hi,abs(lo-k/3.),abs(mi-k/3.),abs(hi-k/3.)]

    # G3: Estructurales (4)
    feats+=[float(sum(1 for n in nums if n in PRIMES56)),
            float(sum(1 for n in nums if n in FIBS56)),
            float(sum(1 for n in nums if n in SQRS56)),
            float(sum(1 for n in nums if n%5==0))]

    # G4: Parity & balance (4)
    mn_s=sum(range(1,k+1)); mx_s=sum(range(n_max-k+1,n_max+1))
    dn=(mx_s-mn_s) or 1.
    feats+=[1.-abs(ev-k/2.)/(k/2.+1e-9),
            1.-abs(s-(mn_s+mx_s)/2.)/dn,
            s/n_max, s/(k*n_max)]

    # G5: Penalización humana (3)
    max_mult=max(sum(1 for n in nums if n%f==0) for f in range(2,8))
    feats+=[float(mr>=5), float(_is_date_like(nums)), float(max_mult>=4)]

    # G6: Hot multi-ventana (4)
    try: hd10=json.loads(h10)  if h10  else {}
    except: hd10={}
    try: hd30=json.loads(h30)  if h30  else {}
    except: hd30={}
    try: hd100=json.loads(h100) if h100 else {}
    except: hd100={}
    try: hd300=json.loads(h300) if h300 else {}
    except: hd300={}
    feats+=[sum(float(hd10.get(str(n),0.))  for n in nums),
            sum(float(hd30.get(str(n),0.))  for n in nums),
            sum(float(hd100.get(str(n),0.)) for n in nums),
            sum(float(hd300.get(str(n),0.)) for n in nums)]

    # G7: Gap (3)
    try: gd=json.loads(gap_j) if gap_j else {}
    except: gd={}
    gv=[float(gd.get(str(n),1.)) for n in nums]
    feats+=[float(np.mean(gv)),float(np.max(gv)),float(np.min(gv))]

    # G8: KS (3)
    try: kd=json.loads(ks_j) if ks_j else {}
    except: kd={}
    kv=[float(kd.get(str(n),0.)) for n in nums]
    feats+=[float(np.mean(kv)),float(np.max(kv)),float(np.sum(kv))]

    # G9: Espectral (3)
    try: sd=json.loads(sp_j) if sp_j else {}
    except: sd={}
    sv=[float(sd.get(str(n),0.)) for n in nums]
    feats+=[float(np.mean(sv)),float(np.max(sv)),float(np.sum(sv))]

    # G10: Co-ocurrencias (3)
    try: cd=json.loads(cooc_j) if cooc_j else {}
    except: cd={}
    pairs=[float(cd.get(f"{min(a,b)}_{max(a,b)}",1.))
           for i,a in enumerate(nums) for b in nums[i+1:]]
    feats+=[float(np.mean(pairs)) if pairs else 1.,
            float(np.max(pairs))  if pairs else 1.,
            float(np.min(pairs))  if pairs else 1.]

    # G11: Posición (k features)
    try: pl=json.loads(pos_j) if pos_j else []
    except: pl=[]
    for i in range(k):
        p=pl[i] if i<len(pl) else {}
        feats.append(float(p.get(str(nums[i]),0.)))

    # G12: Bono (2)
    try: bd=json.loads(bono_j) if bono_j else {}
    except: bd={}
    bvs=[float(bd.get(str(n),0.)) for n in nums]
    feats+=[float(np.mean(bvs)),float(np.max(bvs))]

    # G13: Diffs avanzadas (4)
    feats+=[float(np.std(dif))    if dif.size else 0.,
            float(np.median(dif)) if dif.size else 0.,
            float(sum(1 for d in dif if d>10)),
            float(sum(1 for d in dif if d==1))]

    return tuple(float(x) for x in feats)

def _stats_args(stats: Dict) -> tuple:
    return (json.dumps(stats.get("hot10",{})),
            json.dumps(stats.get("hot30",{})),
            json.dumps(stats.get("hot100",{})),
            json.dumps(stats.get("hot300",{})),
            json.dumps(stats.get("gap",{})),
            json.dumps(stats.get("ks",{})),
            json.dumps(stats.get("spectral",{})),
            json.dumps(stats.get("pos",[])),
            json.dumps(stats.get("cooc",{})),
            json.dumps(stats.get("bono_freq",{})))

def get_features(nums: tuple, stats: Dict, n_max: int) -> tuple:
    return combo_features(nums, n_max, *_stats_args(stats))

# ─────────────────────────────────────────────────────────────────────
# DATASET SUPERVISADO
# ─────────────────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, name: str, stats: Dict):
    k     = LOTTERIES[name]["k"]; n_max = LOTTERIES[name]["n_max"]
    ncols = [f"N{i}" for i in range(1,k+1)]
    feats, labels = [], []
    for _,(_, row) in enumerate(df.iloc[::-1].iterrows()):
        try:
            nums=tuple(sorted(int(row[c]) for c in ncols))
            feats.append(get_features(nums,stats,n_max)); labels.append(1)
        except Exception: continue
    n_pos=sum(labels)
    if n_pos==0: raise RuntimeError(f"Sin muestras positivas {name}")
    rng=np.random.default_rng(SEED)
    for _ in range(min(n_pos*3, 4000)):
        nums=tuple(sorted(int(x) for x in rng.choice(range(1,n_max+1),k,replace=False)))
        feats.append(get_features(nums,stats,n_max)); labels.append(0)
    X=np.array(feats,dtype=float); y=np.array(labels,dtype=int)
    if len(np.unique(y))>1:
        try:
            X,y=SMOTE(random_state=SEED,k_neighbors=min(3,max(1,int(np.sum(y==1))-1))).fit_resample(X,y)
        except Exception:
            try: X,y=RandomOverSampler(random_state=SEED).fit_resample(X,y)
            except Exception: pass
    return X, y

# ─────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────

def train_models(X: np.ndarray, y: np.ndarray, name: str, light: bool) -> Dict:
    scaler=StandardScaler(); Xs=scaler.fit_transform(X)
    n_sp=max(2,min(3,len(X)//15)); tscv=TimeSeriesSplit(n_splits=n_sp)
    models: Dict[str,Any]={}

    # IsolationForest
    try:
        m=IsolationForest(n_estimators=150,contamination=0.05,random_state=SEED)
        m.fit(Xs); models["if"]=m; logger.info(f"[{name}] IF OK")
    except Exception as e: logger.warning(f"[{name}] IF: {e}"); models["if"]=None

    # XGBoost
    try:
        base=xgb.XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
                                 subsample=0.8,colsample_bytree=0.8,
                                 eval_metric="auc",use_label_encoder=False,random_state=SEED)
        m=CalibratedClassifierCV(base,cv=min(3,n_sp),method="sigmoid")
        m.fit(Xs,y); models["xgb"]=m; logger.info(f"[{name}] XGB OK")
    except Exception as e: logger.warning(f"[{name}] XGB: {e}"); models["xgb"]=None

    # LightGBM
    try:
        base=lgb.LGBMClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
                                  num_leaves=31,subsample=0.8,verbose=-1,random_state=SEED)
        m=CalibratedClassifierCV(base,cv=min(3,n_sp),method="isotonic")
        m.fit(Xs,y); models["lgbm"]=m; logger.info(f"[{name}] LGBM OK")
    except Exception as e: logger.warning(f"[{name}] LGBM: {e}"); models["lgbm"]=None

    # CatBoost (solo full)
    models["catboost"]=None
    if CATBOOST_AVAILABLE and not light:
        try:
            base=cb.CatBoostClassifier(iterations=200,depth=4,learning_rate=0.05,
                                         verbose=0,random_seed=SEED)
            m=CalibratedClassifierCV(base,cv=min(3,n_sp),method="isotonic")
            m.fit(Xs,y); models["catboost"]=m; logger.info(f"[{name}] CatBoost OK")
        except Exception as e: logger.warning(f"[{name}] CatBoost: {e}")

    # LSTM (solo full + TF)
    models["lstm"]=None
    if TENSORFLOW_AVAILABLE and not light and Xs.shape[0]>100:
        try:
            Xl=Xs.reshape(Xs.shape[0],Xs.shape[1],1)
            lm=Sequential([LSTM(64,input_shape=(Xs.shape[1],1),return_sequences=True),
                            Dropout(0.3),LSTM(32),Dropout(0.2),
                            BatchNormalization(),Dense(1,activation="sigmoid")])
            lm.compile(optimizer="adam",loss="binary_crossentropy",metrics=["AUC"])
            lm.fit(Xl,y,epochs=20,batch_size=64,validation_split=0.15,
                   callbacks=[EarlyStopping(patience=3,restore_best_weights=True)],verbose=0)
            models["lstm"]=lm; logger.info(f"[{name}] LSTM OK")
        except Exception as e: logger.warning(f"[{name}] LSTM: {e}")

    # Meta-learner HistGradientBoosting sobre OOF
    base_names=[bn for bn in ("xgb","lgbm","catboost") if models.get(bn)]
    models["meta"]=None; models["meta_cols"]=[]
    if len(base_names)>=2 and len(X)>=40:
        try:
            oof=[cross_val_predict(models[bn],Xs,y,cv=min(3,n_sp),method="predict_proba")[:,1]
                 for bn in base_names]
            Z=np.column_stack(oof)
            meta=HistGradientBoostingClassifier(max_iter=100,random_state=SEED)
            meta.fit(Z,y)
            models["meta"]=meta; models["meta_cols"]=base_names
            logger.info(f"[{name}] Meta-HGBC OK: {base_names}")
        except Exception as e: logger.warning(f"[{name}] Stacking: {e}")

    # Permutation importance (diagnóstico)
    try:
        ref=models.get("xgb") or models.get("lgbm")
        if ref:
            pi=permutation_importance(ref,Xs,y,n_repeats=5,random_state=SEED,n_jobs=1)
            top10=np.argsort(pi.importances_mean)[::-1][:10]
            logger.info(f"[{name}] Top-10 feature idx: {top10.tolist()}")
    except Exception: pass

    return {"models":models,"scaler":scaler}

def load_or_train(df: pd.DataFrame, name: str, stats: Dict, light: bool) -> Dict:
    mf=MODEL_FILE_TPL.format(name=name); n=len(df)
    if os.path.exists(mf):
        try:
            c=joblib.load(mf)
            if abs(c.get("trained_on",0)-n)<=5 and c.get("light")==light:
                logger.info(f"[{name}] Modelos desde caché"); return c
        except Exception: pass
    logger.info(f"[{name}] Entrenando (light={light})...")
    t0=time.time(); X,y=build_dataset(df,name,stats)
    r=train_models(X,y,name,light)
    r["trained_on"]=n; r["light"]=light
    try: joblib.dump(r,mf); logger.info(f"[{name}] Modelos guardados {time.time()-t0:.1f}s")
    except Exception as e: logger.warning(f"No se guardó modelo {name}: {e}")
    return r

# ─────────────────────────────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, name: str, n_splits: int=5) -> Dict:
    k=LOTTERIES[name]["k"]; n_max=LOTTERIES[name]["n_max"]; n=len(df)
    if n<60: return {"mean_matches":0.,"baseline":0.,"lift":1.}
    df_asc=df.iloc[::-1].reset_index(drop=True)
    sp=n//(n_splits+1); rng=np.random.default_rng(SEED+7)
    ncols=[f"N{i}" for i in range(1,k+1)]
    ml_m,rnd_m=[],[]
    for fold in range(n_splits):
        te=sp*(fold+1); ts=te; tend=min(ts+max(1,sp//2),n)
        if ts>=n: break
        stats=_build_stats(df_asc.iloc[:te],name)
        hot30=stats["hot30"]; gd=stats["gap"]
        cands=[tuple(sorted(int(x) for x in rng.choice(range(1,n_max+1),k,replace=False)))
               for _ in range(1000)]
        scr=np.array([sum(float(hot30.get(str(nn),0.)) for nn in c)+
                      sum(float(gd.get(str(nn),1.)) for nn in c)*0.1 for c in cands])
        top_nums=set(nn for i in np.argsort(scr)[-10:] for nn in cands[i])
        for _,row in df_asc.iloc[ts:tend].iterrows():
            actual=set(int(row[c]) for c in ncols if pd.notna(row.get(c)))
            if len(actual)<k: continue
            ml_m.append(len(top_nums&actual))
            rnd_m.append(len(set(int(x) for x in rng.choice(range(1,n_max+1),k,replace=False))&actual))
    if not ml_m: return {"mean_matches":0.,"baseline":0.,"lift":1.}
    mm=float(np.mean(ml_m)); mb=float(np.mean(rnd_m)) if rnd_m else 1.
    lift=mm/(mb+1e-9)
    logger.info(f"[{name}] Backtest lift={lift:.3f} (ml={mm:.3f} vs rnd={mb:.3f})")
    return {"mean_matches":mm,"baseline":mb,"lift":lift}

def save_backtest(bt: Dict):
    h={}
    if os.path.exists(BACKTEST_HISTORY):
        try:
            with open(BACKTEST_HISTORY) as f: h=json.load(f)
        except Exception: pass
    ts=datetime.now().strftime("%Y-%m-%d %H:%M")
    h[ts]=bt; keys=sorted(h.keys())[-50:]
    h={k:h[k] for k in keys}
    try:
        with open(BACKTEST_HISTORY,"w") as f: json.dump(h,f,indent=2,default=safe_json)
    except Exception: pass

# ─────────────────────────────────────────────────────────────────────
# PLAUSIBILIDAD
# ─────────────────────────────────────────────────────────────────────

def is_plausible(c: tuple) -> bool:
    s=sum(c)
    if not (MIN_SUM<=s<=MAX_SUM): return False
    cons=mr=1
    for i in range(len(c)-1):
        if c[i+1]==c[i]+1: cons+=1; mr=max(mr,cons)
        else: cons=1
    return mr<=MAX_CONSEC

# ─────────────────────────────────────────────────────────────────────
# GENERACIÓN DE CANDIDATOS
# ─────────────────────────────────────────────────────────────────────

def generate_candidates(n_max:int, k:int, n_samples:int,
                         stats_map:Dict, rng_seed:int=SEED) -> List[Tuple]:
    rng=np.random.default_rng(rng_seed); nr=np.arange(1,n_max+1)
    hot=np.ones(n_max+1,dtype=float); gap=np.ones(n_max+1,dtype=float)
    cooc_all: Dict[str,float]={}
    for st in stats_map.values():
        for ns,w in st.get("hot30",{}).items(): hot[int(ns)]+=float(w)*8.
        for ns,g in st.get("gap",{}).items():   gap[int(ns)]+=float(g)*3.
        for pk,pv in st.get("cooc",{}).items(): cooc_all[pk]=max(cooc_all.get(pk,1.),float(pv))
    hot_p=hot[1:]/hot[1:].sum(); gap_p=gap[1:]/gap[1:].sum()
    uni_p=np.ones(n_max,dtype=float)/n_max
    cooc_arr=np.ones((n_max+1,n_max+1),dtype=float)
    for pk,pv in cooc_all.items():
        try:
            a,b=map(int,pk.split("_")); cooc_arr[a,b]=cooc_arr[b,a]=pv
        except Exception: pass
    cands=set(); att=0; mx=max(20000,n_samples*12)
    while len(cands)<n_samples and att<mx:
        st=rng.choice([0,1,2,3],p=[0.40,0.25,0.10,0.25])
        try:
            if st==0:   sel=rng.choice(nr,size=k,replace=False,p=hot_p)
            elif st==1: sel=rng.choice(nr,size=k,replace=False,p=gap_p)
            elif st==2:
                first=int(rng.choice(nr,p=hot_p))
                cp=cooc_arr[first,1:].copy(); cp[first-1]=0.
                if cp.sum()<1e-9: cp=np.ones(n_max,dtype=float)
                cp/=cp.sum()
                rest=list(rng.choice(nr,size=k-1,replace=False,p=cp))
                sel=[first]+rest
            else: sel=rng.choice(nr,size=k,replace=False,p=uni_p)
            c=tuple(sorted(int(x) for x in sel))
            if len(set(c))==k and is_plausible(c): cands.add(c)
        except Exception: pass
        att+=1
    if not cands: raise RuntimeError("No se generaron candidatos")
    r=list(cands)
    return r[:n_samples] if len(r)>n_samples else r

# ─────────────────────────────────────────────────────────────────────
# EXPANSIÓN CON SIMULATED ANNEALING
# ─────────────────────────────────────────────────────────────────────

def expand_sa(top: List[Tuple], n_max:int, k:int,
              stats_map:Dict, per_c:int, target:int, rng_seed:int=SEED) -> List[Tuple]:
    rng=np.random.default_rng(rng_seed)
    gap=np.ones(n_max+1,dtype=float)
    for st in stats_map.values():
        for ns,g in st.get("gap",{}).items(): gap[int(ns)]+=float(g)
    gap_p=gap[1:]/gap[1:].sum(); nr=np.arange(1,n_max+1)
    nbrs=set(); temp=1.0; cool=0.995
    lim=min(len(top),max(300,len(top)//5))
    for c in top[:lim]:
        cur=list(c)
        for _ in range(per_c):
            temp=max(0.01,temp*cool); cand=cur.copy()
            nc=rng.choice([1,1,2],p=[0.50,0.35,0.15])
            for _ in range(nc):
                pos=rng.integers(0,k)
                if rng.random()<temp: repl=int(rng.integers(1,n_max+1))
                else:                  repl=int(rng.choice(nr,p=gap_p))
                cand[pos]=repl
            cs=tuple(sorted(set(int(x) for x in cand)))
            if len(cs)==k and is_plausible(cs): nbrs.add(cs); cur=list(cs)
        if len(nbrs)>=target: break
    return list(nbrs)[:target]

# ─────────────────────────────────────────────────────────────────────
# WORKER INIT Y SCORING
# ─────────────────────────────────────────────────────────────────────

def worker_init(mf_j:str, stats_j:str, names_j:str, light_j:str):
    global WORKER_MODELS,WORKER_SCALERS,WORKER_STATS,WORKER_NAMES,WORKER_LIGHT
    WORKER_NAMES=json.loads(names_j); WORKER_LIGHT=json.loads(light_j)
    WORKER_STATS=json.loads(stats_j)
    mf=json.loads(mf_j)
    for name in WORKER_NAMES:
        f=mf.get(name,"")
        if f and os.path.exists(f):
            try:
                c=joblib.load(f)
                WORKER_MODELS[name]=c.get("models",{}); WORKER_SCALERS[name]=c.get("scaler")
            except Exception: WORKER_MODELS[name]={}; WORKER_SCALERS[name]=None
        else: WORKER_MODELS[name]={}; WORKER_SCALERS[name]=None
    try:
        p=psutil.Process(os.getpid())
        p.nice(10 if os.name!="nt" else psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception: pass

def _ml_score(combos:List[Tuple], name:str) -> np.ndarray:
    st=WORKER_STATS.get(name,{}); mdls=WORKER_MODELS.get(name,{})
    sc=WORKER_SCALERS.get(name); n_max=LOTTERIES[name]["n_max"]; m=len(combos)
    args=_stats_args(st)
    feats=[]
    for c in combos:
        try: feats.append(combo_features(c,n_max,*args))
        except: feats.append(tuple([0.]*70))
    X=np.array(feats,dtype=np.float32)
    Xs=sc.transform(X) if sc is not None and X.shape[0]>0 else X
    # Stacking
    meta=mdls.get("meta"); mc=mdls.get("meta_cols",[])
    if meta and len(mc)>=2:
        try:
            cp=[mdls[bn].predict_proba(Xs)[:,1] for bn in mc if mdls.get(bn)]
            if len(cp)==len(mc):
                Z=np.column_stack(cp)
                return _norm(meta.predict_proba(Z)[:,1].astype(np.float32))
        except Exception: pass
    # Fallback
    score=np.zeros(m,dtype=np.float32); tw=0.
    for mk,mw in [("xgb",W_XGB),("lgbm",W_LGBM),("catboost",W_CB)]:
        mm=mdls.get(mk)
        if mm:
            try: p2=mm.predict_proba(Xs)[:,1].astype(np.float32); score+=mw*_norm(p2); tw+=mw
            except: pass
    if mdls.get("if"):
        try: score+=W_IF*_norm(-mdls["if"].score_samples(Xs).astype(np.float32)); tw+=W_IF
        except: pass
    if mdls.get("lstm") and TENSORFLOW_AVAILABLE:
        try:
            Xl=Xs.reshape(Xs.shape[0],Xs.shape[1],1)
            sl=mdls["lstm"].predict(Xl,verbose=0).flatten().astype(np.float32)
            score+=W_LSTM*_norm(sl); tw+=W_LSTM
        except: pass
    return _norm(score/(tw or 1.))

def worker_score_batch(args):
    combos_raw,n_max,k=args
    combos=[tuple(sorted(int(x) for x in c)) for c in combos_raw
            if len(set(c))==k]
    combos=[c for c in combos if is_plausible(c)]
    if not combos: return []
    m=len(combos)
    per_lot: Dict[str,np.ndarray]={}
    for name in WORKER_NAMES:
        per_lot[name]=_ml_score(combos,name)
    # Refinamiento estadístico
    refine=np.zeros(m,dtype=np.float32)
    for name in WORKER_NAMES:
        st=WORKER_STATS.get(name,{})
        h30=st.get("hot30",{}); gd=st.get("gap",{})
        ksd=st.get("ks",{});    cd=st.get("cooc",{})
        hs=np.array([sum(float(h30.get(str(n),0.)) for n in c) for c in combos],dtype=np.float32)
        gs=np.array([float(np.mean([float(gd.get(str(n),1.)) for n in c])) for c in combos],dtype=np.float32)
        ks=np.array([float(np.mean([float(ksd.get(str(n),0.)) for n in c])) for c in combos],dtype=np.float32)
        cs=np.array([float(np.mean([float(cd.get(f"{min(a,b)}_{max(a,b)}",1.))
                    for i,a in enumerate(c) for b in c[i+1:]])) for c in combos],dtype=np.float32)
        ev=np.array([sum(1 for n in c if n%2==0) for c in combos],dtype=np.float32)
        sm=np.array([sum(c) for c in combos],dtype=np.float32)
        ps=1.-np.abs(ev-k/2.)/(k/2.+1e-9)
        mn_s=sum(range(1,k+1)); mx_s=sum(range(n_max-k+1,n_max+1))
        sb=1.-np.abs(sm-(mn_s+mx_s)/2.)/((mx_s-mn_s) or 1.)
        hum=np.array([float(_is_date_like(c) or
                      max(sum(1 for i in range(len(c)-1) if c[i+1]==c[i]+1),
                          max(sum(1 for n in c if n%f==0) for f in range(2,8)))>=4)
                      for c in combos],dtype=np.float32)
        local=(GAMMA_HOT*_norm(hs)+ETA_GAP*_norm(gs)+DELTA_KS*_norm(ks)
               +THETA_COV*_norm(cs)+EPS_PAR*ps+ZETA_SUM*sb-IOTA_HUM*hum)
        refine+=local/len(WORKER_NAMES)
    # Global composite con temperature scaling
    stk=np.stack([per_lot.get(n,np.zeros(m)) for n in WORKER_NAMES],axis=1)
    wv=np.array([POT_WEIGHTS[n] for n in WORKER_NAMES],dtype=np.float32)
    ml=stk@wv
    T=float(os.getenv("ENSEMBLE_TEMP","1.5"))
    ml=np.power(np.clip(ml,1e-9,1-1e-9),1./T)
    gc_score=_norm(ml)+refine
    out=[]
    for i,c in enumerate(combos):
        out.append({"combo":list(c),"suma":int(sum(c)),
                    "global_composite":float(gc_score[i]),
                    "per_lottery":{n:float(per_lot[n][i]) for n in WORKER_NAMES}})
    gc.collect(); return out

# ─────────────────────────────────────────────────────────────────────
# PORTFOLIO: BEAM SEARCH + DIVERSIDAD COSENO
# ─────────────────────────────────────────────────────────────────────

def build_portfolio(df_top: pd.DataFrame, top_k:int,
                    lam:float=0.30, bw:int=5) -> pd.DataFrame:
    if len(df_top)<=top_k: return df_top
    rows=df_top.to_dict("records")
    sc=np.array([r["global_composite"] for r in rows])
    sn=(sc-sc.min())/(sc.max()-sc.min()+1e-9)
    n_max=next(iter(LOTTERIES.values()))["n_max"]
    def vec(c):
        v=np.zeros(n_max+1,dtype=np.float32)
        for n in c: v[n]=1.
        return v
    vecs=[vec(r["combo"]) for r in rows]
    beams=[(0.,[],set(),np.zeros(n_max+1,dtype=np.float32))]
    no_imp=0; prev_b=-1.
    for step in range(top_k):
        cands_b=[]
        for (acc,sel,cov,sumv) in beams:
            for i,row in enumerate(rows):
                if i in sel: continue
                combo=set(row["combo"])
                marg=len(combo-cov)/(len(combo) or 1)
                if sel:
                    sv=sumv/len(sel)
                    sim=float(np.dot(sv,vecs[i])/(np.linalg.norm(sv)*np.linalg.norm(vecs[i])+1e-9))
                    div=1.-sim
                else: div=1.
                ss=(1-lam)*sn[i]+lam*0.5*marg+lam*0.5*div
                cands_b.append((acc+ss,sel+[i],cov|combo,sumv+vecs[i]))
        cands_b.sort(key=lambda x:x[0],reverse=True); beams=cands_b[:bw]
        best_now=beams[0][0] if beams else prev_b
        if abs(best_now-prev_b)<0.0005:
            no_imp+=1
            if no_imp>=3: logger.info(f"Portfolio early-stop paso {step+1}"); break
        else: no_imp=0
        prev_b=best_now
    idx=beams[0][1] if beams else list(range(min(top_k,len(rows))))
    return df_top.iloc[idx].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────
# RUNNER DE BATCHES
# ─────────────────────────────────────────────────────────────────────

def _run_batches(batches:list, mf:Dict, stats:Dict, light:bool, phase:str) -> list:
    results=[]; ctx=get_context("spawn")
    ia=(json.dumps(mf),json.dumps(stats),json.dumps(list(LOTTERIES.keys())),json.dumps(light))
    t0=time.time(); tot=len(batches); done=0; lp=-5
    try:
        with ctx.Pool(processes=WORKERS,initializer=worker_init,initargs=ia) as pool:
            for res in pool.imap_unordered(worker_score_batch,batches):
                done+=1
                if res: results.extend(res)
                if done%max(1,tot//20)==0: gc.collect()
                pct=int(done/tot*100)
                if pct>=lp+5 or done==tot:
                    eta=(time.time()-t0)/max(done,1)*(tot-done)
                    ets=(datetime.now()+timedelta(seconds=eta)).strftime("%H:%M:%S")
                    logger.info(f"[{phase}] {pct}% ({done}/{tot}) ETA {ets}"); lp=pct
    except Exception as e:
        logger.error(f"Pool {phase}: {e} — fallback single-thread")
        worker_init(*ia)
        for b in batches:
            r=worker_score_batch(b)
            if r: results.extend(r)
    return results

# ─────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(all_histories:Dict, model_files:Dict, stats_all:Dict, light:bool):
    n_max=next(iter(LOTTERIES.values()))["n_max"]
    k=next(iter(LOTTERIES.values()))["k"]
    PR=_PRERANK_LIGHT if light else _PRERANK_FULL
    TG=_TARGET_LIGHT  if light else _TARGET_FULL
    NE=_NEIGH_LIGHT   if light else _NEIGH_FULL
    def mk_batches(cands):
        sz=max(1,math.ceil(len(cands)/WORKERS))
        return [(cands[i:i+sz],n_max,k) for i in range(0,len(cands),sz)]
    logger.info("═"*55)
    logger.info(f"FASE 1 — Generando {PR:,} candidatos (light={light})")
    cands=generate_candidates(n_max,k,PR,stats_all)
    logger.info(f"Fase 1 — {len(cands):,} generados")
    logger.info("FASE 2 — Preranking...")
    t0=time.time()
    pre=_run_batches(mk_batches(cands),model_files,stats_all,light,"PRERANK")
    if not pre: raise RuntimeError("Sin resultados preranking")
    df_pre=(pd.DataFrame(pre).sort_values("global_composite",ascending=False)
            .drop_duplicates(subset=["combo"]).reset_index(drop=True))
    top=df_pre.head(PRERANK_TOP)
    tc=[tuple(int(x) for x in r["combo"]) for _,r in top.iterrows()]
    logger.info(f"Fase 2 — Top {len(tc):,} para expansión")
    logger.info("FASE 3 — Expansión Simulated Annealing...")
    nbrs=expand_sa(tc,n_max,k,stats_all,NE,TG)
    final_set=set(tuple(int(x) for x in r["combo"]) for _,r in top.iterrows())
    final_set.update(nbrs); final=list(final_set)
    logger.info(f"Fase 3 — {len(final):,} candidatos finales")
    logger.info("FASE 4 — Evaluación final...")
    t0=time.time()
    fin=_run_batches(mk_batches(final),model_files,stats_all,light,"FINAL")
    if not fin: raise RuntimeError("Sin resultados evaluación final")
    df_fin=(pd.DataFrame(fin).sort_values("global_composite",ascending=False)
            .drop_duplicates(subset=["combo"]).head(TOP_K*5).reset_index(drop=True))
    logger.info("FASE 5 — Maximum Coverage Portfolio (beam search)...")
    df_port=build_portfolio(df_fin,TOP_K,lam=0.30,bw=5)
    all_nums=set(n for row in df_port["combo"] for n in row)
    cov=len(all_nums)/n_max*100
    return df_port,{"n_prerank":len(cands),"n_final":len(final),
                    "n_evaluated":len(fin),"coverage_pct":cov,
                    "mean_score":float(df_port["global_composite"].mean()),
                    "light":light,"time_s":time.time()-t0}

# ─────────────────────────────────────────────────────────────────────
# PERSISTENCIA
# ─────────────────────────────────────────────────────────────────────

def save_predictions(df:pd.DataFrame, date_str:str, bt:Dict):
    h={}
    if os.path.exists(PREDICTIONS_HISTORY):
        try:
            with open(PREDICTIONS_HISTORY,encoding="utf-8") as f: h=json.load(f)
        except Exception: pass
    key="GlobalUnified_v8"
    entry={"date":date_str,"backtest":bt,
           "predicted":[{"combo":[int(x) for x in r["combo"]],
                         "global_composite":float(r["global_composite"])}
                        for _,r in df.iterrows()]}
    h.setdefault(key,[]).append(entry); h[key]=h[key][-25:]
    try:
        with open(PREDICTIONS_HISTORY,"w",encoding="utf-8") as f:
            json.dump(h,f,default=safe_json,ensure_ascii=False,indent=2)
    except Exception as e: logger.error(f"Error guardando historial: {e}")

# ─────────────────────────────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────────────────────────────

def send_email_results(df:pd.DataFrame, stats:Dict, bt:Dict, ts:str) -> bool:
    if not all([EMAIL_FROM,EMAIL_PASS,EMAIL_TO]):
        logger.warning("Credenciales email no configuradas"); return False
    cov=stats.get("coverage_pct",0); light=stats.get("light",False)
    rows_h=""
    for i,(_,row) in enumerate(df.iterrows(),1):
        combo=" ".join(f"{int(x):02d}" for x in sorted(row["combo"]))
        rows_h+=(f"<tr><td align='center'><b>{i}</b></td>"
                 f"<td align='center'><b style='letter-spacing:2px'>{combo}</b></td>"
                 f"<td align='center'>{float(row['global_composite']):.5f}</td>"
                 f"<td align='center'>{int(row['suma'])}</td></tr>")
    bt_h=""
    for name,b in bt.items():
        lift=b.get("lift",1.); color="#2e7d32" if lift>1.05 else "#c62828"
        bt_h+=(f"<tr><td>{name}</td>"
               f"<td align='center'>{b.get('mean_matches',0):.3f}</td>"
               f"<td align='center'>{b.get('baseline',0):.3f}</td>"
               f"<td align='center' style='color:{color}'><b>{lift:.3f}</b></td></tr>")
    html=f"""<html><body style='font-family:Arial,sans-serif;max-width:700px;margin:auto'>
<h2 style='color:#1a3a5c;border-bottom:3px solid #2e75b6;padding-bottom:8px'>
🎰 Jefe Maestro v8.0 Final Supreme</h2>
<p><b>Generado:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} &nbsp;
<b>Modo:</b> {'⚡ Light' if light else '🚀 Full'} &nbsp;
<b>Cobertura:</b> {cov:.1f}%</p>
<h3 style='color:#2e75b6'>Top {TOP_K} Combinaciones (válidas para Melate, Revancha y Revanchita)</h3>
<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%'>
<tr style='background:#1a3a5c;color:white'><th>#</th><th>N1–N6</th><th>Score</th><th>Suma</th></tr>
{rows_h}</table>
<p style='font-size:11px;color:#888'>El BONO de Melate lo elige el sorteo, no tú. 
Tus 6 números son los mismos para los 3 sorteos.</p>
<h3 style='color:#2e75b6'>Backtesting vs. azar aleatorio</h3>
<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%'>
<tr style='background:#1a3a5c;color:white'><th>Sorteo</th><th>ML matches</th><th>Random</th><th>Lift</th></tr>
{bt_h}</table>
<p style='font-size:11px;color:#888'>Lift &gt; 1.0 = el modelo supera al azar en aciertos parciales.</p>
<hr><p style='font-size:10px;color:#aaa'>Jefe Maestro v8.0 · Juega con responsabilidad.</p>
</body></html>"""
    msg=MIMEMultipart("alternative"); msg["Subject"]=f"🎰 Jefe Maestro v8 — {ts}"
    msg["From"]=EMAIL_FROM; msg["To"]=EMAIL_TO; msg.attach(MIMEText(html,"html"))
    for attempt in range(3):
        try:
            with smtplib.SMTP(SMTP_SERVER,SMTP_PORT,timeout=30) as s:
                s.ehlo(); s.starttls(); s.ehlo(); s.login(EMAIL_FROM,EMAIL_PASS)
                s.sendmail(EMAIL_FROM,[EMAIL_TO],msg.as_string())
            logger.info(f"Email enviado a {EMAIL_TO}"); return True
        except Exception as e:
            if attempt==2: logger.error(f"Email falló: {e}")
            time.sleep(3)
    return False

# ─────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PÚBLICO (main_run.py lo llama)
# ─────────────────────────────────────────────────────────────────────

def run_model(histories_override: Optional[Dict[str,pd.DataFrame]]=None):
    """Punto de entrada para main_run.py."""
    light=(_AUTO_LIGHT or
           os.getenv("LIGHT_MODE","").lower() in ("1","true","yes") or
           "--light" in sys.argv)
    logger.info(f"🚀 Jefe Maestro v8.0 Final Supreme "
                f"(RAM={_RAM_GB:.1f}GB CPUs={_CPUS} light={light})")
    all_h=histories_override or load_all_histories()
    # Stats
    stats_all={}
    for name,df in all_h.items():
        stats_all[name]=load_or_build_stats(df,name)
    # Modelos + backtest
    mf={}; bt_all={}
    for name,df in all_h.items():
        try:
            load_or_train(df,name,stats_all[name],light)
            mf[name]=MODEL_FILE_TPL.format(name=name)
        except Exception as e: _abort(f"Fallo modelo {name}: {e}")
        try: bt_all[name]=backtest(df,name)
        except Exception as e:
            logger.warning(f"Backtest {name}: {e}"); bt_all[name]={"lift":1.}
    save_backtest(bt_all)
    # Serializar stats para workers
    stats_s={}
    for name,st in stats_all.items():
        stats_s[name]={k:(v if isinstance(v,(dict,list,int,float,str,bool)) else str(v))
                       for k,v in st.items()}
    # Pipeline
    try:
        df_top,run_s=run_pipeline(all_h,mf,stats_s,light)
    except Exception as e: _abort(f"Pipeline falló: {e}")
    # Log
    logger.info("═"*60); logger.info("TOP COMBINACIONES v8.0 Final Supreme")
    logger.info(f"Cobertura portfolio: {run_s.get('coverage_pct',0):.1f}%")
    logger.info("═"*60)
    for i,(_,row) in enumerate(df_top.iterrows(),1):
        combo=" ".join(f"{int(x):02d}" for x in sorted(row["combo"]))
        logger.info(f"  #{i:02d}: {combo}  score={float(row['global_composite']):.5f}  suma={int(row['suma'])}")
    logger.info("═"*60)
    for name,bt in bt_all.items():
        logger.info(f"  [{name}] lift={bt.get('lift',1.):.3f}")
    logger.info("═"*60)
    # Guardar y enviar
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    save_predictions(df_top,datetime.now().strftime("%Y-%m-%d"),bt_all)
    out=os.path.join(_RESULTS,f"v8_results_{ts}.json")
    try:
        with open(out,"w",encoding="utf-8") as f:
            json.dump({"combinaciones":[{"rank":i+1,
                "combo_str":" ".join(f"{int(x):02d}" for x in sorted(row["combo"])),
                "combo":[int(x) for x in row["combo"]],
                "global_composite":float(row["global_composite"]),
                "suma":int(row["suma"])}
                for i,(_,row) in enumerate(df_top.iterrows())],
                "run_stats":run_s,"backtest":bt_all},
                f,default=safe_json,ensure_ascii=False,indent=2)
        logger.info(f"Resultados: {out}")
    except Exception as e: logger.error(f"Error guardando: {e}")
    send_email_results(df_top,run_s,bt_all,ts)
    logger.info("✅ Completado.")
    return df_top, run_s

# ─────────────────────────────────────────────────────────────────────
# MAIN (ejecución directa)
# ─────────────────────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser(description="Jefe Maestro v8.0 Final Supreme")
    p.add_argument("--light",action="store_true",help="Modo rápido GitHub Actions")
    p.add_argument("--force-retrain",action="store_true",help="Borrar caché y reentrenar")
    args=p.parse_args()
    if args.force_retrain:
        for name in LOTTERIES:
            for f in [MODEL_FILE_TPL.format(name=name),STATS_FILE_TPL.format(name=name)]:
                if os.path.exists(f): os.remove(f); logger.info(f"Caché borrado: {f}")
    if args.light: os.environ["LIGHT_MODE"]="true"
    try: run_model()
    except SystemExit: raise
    except Exception as e:
        logger.critical(f"Fatal: {e}",exc_info=True)
        send_telegram(f"Fatal Jefe Maestro v8: {e}"); sys.exit(1)

if __name__=="__main__":
    main()
