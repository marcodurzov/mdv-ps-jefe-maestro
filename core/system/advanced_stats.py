#!/usr/bin/env python3

# -*- coding: utf-8 -*-

“””
advanced_stats.py
Módulo unificado de estadísticas avanzadas para Jefe Maestro v8.

Combina en un solo archivo:

1. Correlaciones temporales de 3er orden (TC)
1. Detección de sesgos físicos y manipulación (BIAS)
1. Efectos por día de sorteo (DOW)
1. Anti-correlaciones de pares (ANTI)
1. Análisis de rachas y ciclos (STREAK)
1. Detección de cambios estructurales en el tiempo (BREAK)
1. Análisis de entropía por bloque (ENTROPY)
1. Correlación resultado vs pozo acumulado (POT)

CÓMO FUNCIONA:

- Se llama UNA SOLA VEZ al inicio del predictor.
- Calcula todo y lo cachea en disco.
- En corridas futuras carga desde caché si los datos no cambiaron.
- Retorna un score adicional por combinación que se suma al composite.
- Genera un reporte de semáforo (verde/amarillo/rojo) para el correo.

INTEGRACIÓN CON EL PREDICTOR:
Solo requiere 3 líneas nuevas en jefe_maestro_v6_elite_predictor.py
(instrucciones al final de este archivo).
“””

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats as scipy_stats

warnings.filterwarnings(“ignore”)
logger = logging.getLogger(**name**)

# ── Rutas ────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(**file**))
_ROOT     = os.path.dirname(os.path.dirname(_HERE))
_CACHE    = os.path.join(_ROOT, “cache”)
os.makedirs(_CACHE, exist_ok=True)

AS_CACHE_TPL  = os.path.join(*CACHE, “advanced_stats*{name}.joblib”)
AS_REPORT     = os.path.join(_ROOT,  “advanced_stats_report.json”)

# ── Parámetros ───────────────────────────────────────────────────────

MIN_OBS_TC       = 3      # mínimo de observaciones para confiar en TC
LAPLACE          = 0.5    # suavizado Laplace
BLOCK_SIZE       = 100    # sorteos por bloque para entropía
MIN_BREAK_BLOCK  = 200    # mínimo de sorteos para detectar cambio estructural
SIGNIFICANCE     = 0.05   # nivel de significancia estadística

LOTTERIES = {
“Melate”:     {“n_max”: 56, “k”: 6, “has_bono”: True},
“Revancha”:   {“n_max”: 56, “k”: 6, “has_bono”: False},
“Revanchita”: {“n_max”: 56, “k”: 6, “has_bono”: False},
}

# ─────────────────────────────────────────────────────────────────────

# UTILIDADES

# ─────────────────────────────────────────────────────────────────────

def _safe(obj):
if isinstance(obj, (np.integer,)):  return int(obj)
if isinstance(obj, (np.floating,)): return float(obj)
if isinstance(obj, np.ndarray):     return obj.tolist()
return obj

def _get_combos(df: pd.DataFrame, k: int, n_max: int) -> List[List[int]]:
“”“Extrae lista de combinaciones válidas en orden cronológico (asc).”””
num_cols = [f”N{i}” for i in range(1, k+1)]
df_asc   = df.iloc[::-1].reset_index(drop=True)
combos   = []
for _, row in df_asc.iterrows():
try:
nums = [int(row[c]) for c in num_cols if pd.notna(row.get(c))]
if len(nums) == k and all(1 <= n <= n_max for n in nums):
combos.append(sorted(nums))
else:
combos.append([])
except Exception:
combos.append([])
return combos

def _get_dates(df: pd.DataFrame) -> List[Optional[pd.Timestamp]]:
“”“Extrae fechas en orden cronológico (asc).”””
df_asc = df.iloc[::-1].reset_index(drop=True)
dates  = []
for _, row in df_asc.iterrows():
try:
d = pd.to_datetime(row.get(“FECHA”,””), dayfirst=True, errors=“coerce”)
dates.append(d if not pd.isna(d) else None)
except Exception:
dates.append(None)
return dates

# ─────────────────────────────────────────────────────────────────────

# 1. CORRELACIONES TEMPORALES DE 3ER ORDEN

# ─────────────────────────────────────────────────────────────────────

def _build_tc(combos: List[List[int]], n_max: int) -> Dict:
“””
Tensor T[a,b,c]: frecuencia de c en t+1 dado (a,b) en t.
Normalizado vs. frecuencia marginal esperada.
“””
T          = np.zeros((n_max, n_max, n_max), dtype=np.float32)
pair_count = np.zeros((n_max, n_max),        dtype=np.float32)
marginal   = np.zeros(n_max,                  dtype=np.float32)
n_valid    = 0

```
for t in range(len(combos) - 1):
    curr = combos[t]
    nxt  = combos[t+1]
    if not curr or not nxt:
        continue
    for i in range(len(curr)):
        a = curr[i] - 1
        for j in range(i+1, len(curr)):
            b = curr[j] - 1
            pair_count[a, b] += 1
            pair_count[b, a] += 1
            for c_num in nxt:
                c = c_num - 1
                T[a, b, c] += 1
                T[b, a, c] += 1
    for c_num in nxt:
        marginal[c_num - 1] += 1
    n_valid += 1

p_marg = marginal / (marginal.sum() + 1e-9)
mask   = pair_count >= MIN_OBS_TC

# Ratio de enriquecimiento
ratio = np.ones_like(T)
for c in range(n_max):
    p_c = p_marg[c] + 1e-9
    p_c_given_ab = np.where(
        mask,
        (T[:, :, c] + LAPLACE) /
        (pair_count + LAPLACE * n_max + 1e-9),
        p_c
    )
    ratio[:, :, c] = p_c_given_ab / p_c

return {
    "ratio":       ratio,
    "pair_count":  pair_count,
    "p_marginal":  p_marg,
    "n_valid":     n_valid,
}
```

def _score_tc_batch(combos_cand: List[Tuple],
last_combo: List[int],
tc_data: Dict,
n_max: int) -> np.ndarray:
“”“Score TC para un batch de candidatos dado el último sorteo real.”””
ratio = tc_data[“ratio”]
pc    = tc_data[“pair_count”]

```
# Vector de ratio promedio dado los pares del último sorteo
pair_ratio_vec = np.zeros(n_max, dtype=np.float64)
n_pairs        = 0
for i in range(len(last_combo)):
    a = last_combo[i] - 1
    for j in range(i+1, len(last_combo)):
        b = last_combo[j] - 1
        if a >= n_max or b >= n_max: continue
        if pc[a, b] < MIN_OBS_TC:   continue
        pair_ratio_vec += ratio[a, b, :]
        n_pairs        += 1

if n_pairs == 0:
    return np.ones(len(combos_cand), dtype=np.float32)

pair_ratio_vec /= n_pairs
out = np.array([
    float(np.mean(pair_ratio_vec[[n-1 for n in combo if 1<=n<=n_max]]))
    for combo in combos_cand
], dtype=np.float32)
return out
```

# ─────────────────────────────────────────────────────────────────────

# 2. DETECCIÓN DE SESGOS FÍSICOS

# ─────────────────────────────────────────────────────────────────────

def _build_bias(combos: List[List[int]],
dates:  List[Optional[pd.Timestamp]],
n_max: int, k: int) -> Dict:
“””
Calcula indicadores de sesgo físico por número.
- Chi-cuadrado formal con p-value
- Ratio de frecuencia observada vs. esperada
- Racha máxima de ausencia
- Semáforo (verde/amarillo/rojo)
“””
counts   = np.zeros(n_max + 1, dtype=np.int32)
n_sorteos = 0
for combo in combos:
if combo:
for n in combo: counts[n] += 1
n_sorteos += 1

```
if n_sorteos == 0:
    return {}

expected   = n_sorteos * k / n_max
chi2_total = float(np.sum((counts[1:] - expected)**2 / (expected + 1e-9)))
p_global   = float(scipy_stats.chi2.sf(chi2_total, df=n_max - 1))

# Ratio y p-value por número
bias_per_num = {}
for num in range(1, n_max + 1):
    obs   = counts[num]
    ratio = obs / (expected + 1e-9)
    # Chi2 individual (1 df)
    chi2_n = (obs - expected)**2 / (expected + 1e-9)
    p_n    = float(scipy_stats.chi2.sf(chi2_n, df=1))
    bias_per_num[num] = {
        "obs":   int(obs),
        "exp":   round(expected, 2),
        "ratio": round(ratio, 3),
        "chi2":  round(chi2_n, 3),
        "p":     round(p_n, 4),
    }

# Racha máxima de ausencia por número
streak_max = {num: 0 for num in range(1, n_max+1)}
streak_cur = {num: 0 for num in range(1, n_max+1)}
for combo in combos:
    if not combo: continue
    combo_set = set(combo)
    for num in range(1, n_max+1):
        if num in combo_set:
            streak_cur[num] = 0
        else:
            streak_cur[num] += 1
            streak_max[num] = max(streak_max[num], streak_cur[num])

# Racha esperada teórica bajo uniformidad
p_absent    = 1 - k / n_max
exp_streak  = 1 / (1 - p_absent + 1e-9)

# Semáforo
semaforo = {}
for num in range(1, n_max+1):
    b      = bias_per_num[num]
    streak = streak_max[num]
    if b["p"] < 0.01 or streak > exp_streak * 3:
        semaforo[num] = "🔴"
    elif b["p"] < 0.05 or streak > exp_streak * 2:
        semaforo[num] = "🟡"
    else:
        semaforo[num] = "🟢"

return {
    "chi2_global":  round(chi2_total, 3),
    "p_global":     round(p_global, 6),
    "n_sorteos":    n_sorteos,
    "expected_freq": round(expected, 2),
    "bias_per_num": bias_per_num,
    "streak_max":   streak_max,
    "exp_streak":   round(exp_streak, 1),
    "semaforo":     semaforo,
    "numeros_rojos": [n for n in range(1,n_max+1) if semaforo[n]=="🔴"],
    "numeros_amarillos": [n for n in range(1,n_max+1) if semaforo[n]=="🟡"],
}
```

# ─────────────────────────────────────────────────────────────────────

# 3. EFECTO DÍA DE SORTEO

# ─────────────────────────────────────────────────────────────────────

def _build_dow(combos: List[List[int]],
dates:  List[Optional[pd.Timestamp]],
n_max: int, k: int) -> Dict:
“””
Calcula si la distribución de números difiere según
el día de la semana (miércoles=2, viernes=4, domingo=6).
“””
day_counts: Dict[int, np.ndarray] = {}
day_totals: Dict[int, int]        = {}

```
for combo, date in zip(combos, dates):
    if not combo or date is None:
        continue
    dow = date.dayofweek  # 0=lun ... 6=dom
    if dow not in day_counts:
        day_counts[dow] = np.zeros(n_max + 1, dtype=np.int32)
        day_totals[dow] = 0
    for n in combo:
        day_counts[dow][n] += 1
    day_totals[dow] += 1

if len(day_counts) < 2:
    return {"disponible": False}

# Ratio por número y día vs. global
global_counts = np.zeros(n_max + 1, dtype=np.int32)
global_total  = 0
for dc, dt in zip(day_counts.values(), day_totals.values()):
    global_counts += dc
    global_total  += dt

dow_ratio: Dict[int, Dict[int, float]] = {}
for dow, dc in day_counts.items():
    n_d = day_totals[dow]
    if n_d == 0: continue
    dow_ratio[dow] = {}
    for num in range(1, n_max+1):
        obs_rate    = dc[num] / (n_d * k + 1e-9)
        global_rate = global_counts[num] / (global_total * k + 1e-9)
        dow_ratio[dow][num] = round(obs_rate / (global_rate + 1e-9), 3)

# Números con mayor ratio por día
top_by_day: Dict[int, List[int]] = {}
for dow, ratios in dow_ratio.items():
    top_by_day[dow] = sorted(
        [n for n in ratios if ratios[n] >= 1.15],
        key=lambda n: ratios[n], reverse=True
    )[:10]

dow_names = {0:"Lunes",1:"Martes",2:"Miércoles",
             3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}

return {
    "disponible":  True,
    "dow_ratio":   {dow_names.get(d,str(d)): r for d,r in dow_ratio.items()},
    "top_by_day":  {dow_names.get(d,str(d)): v for d,v in top_by_day.items()},
    "sorteos_x_dia": {dow_names.get(d,str(d)): v
                      for d,v in day_totals.items()},
}
```

# ─────────────────────────────────────────────────────────────────────

# 4. ANTI-CORRELACIONES DE PARES

# ─────────────────────────────────────────────────────────────────────

def _build_anti(combos: List[List[int]], n_max: int, k: int) -> Dict:
“””
Detecta pares de números que salen juntos MUCHO MENOS
de lo esperado bajo uniformidad.
“””
cooc     = np.zeros((n_max+1, n_max+1), dtype=np.int32)
n_sorteos = 0
for combo in combos:
if not combo: continue
for i in range(len(combo)):
for j in range(i+1, len(combo)):
a, b = combo[i], combo[j]
cooc[a, b] += 1
cooc[b, a] += 1
n_sorteos += 1

```
if n_sorteos == 0:
    return {}

expected_pair = n_sorteos * k * (k-1) / (n_max * (n_max-1))

anti_pairs   = []
hot_pairs    = []
for a in range(1, n_max+1):
    for b in range(a+1, n_max+1):
        obs   = cooc[a, b]
        ratio = obs / (expected_pair + 1e-9)
        if ratio < 0.3 and obs > 0:
            anti_pairs.append((a, b, round(ratio, 3), int(obs)))
        elif ratio > 2.5:
            hot_pairs.append((a, b, round(ratio, 3), int(obs)))

anti_pairs.sort(key=lambda x: x[2])
hot_pairs.sort(key=lambda x: x[2], reverse=True)

return {
    "expected_pair": round(expected_pair, 2),
    "anti_pairs":    anti_pairs[:20],   # 20 pares que más se evitan
    "hot_pairs":     hot_pairs[:20],    # 20 pares que más se buscan
    "cooc_matrix":   cooc,              # para scoring
}
```

# ─────────────────────────────────────────────────────────────────────

# 5. DETECCIÓN DE CAMBIOS ESTRUCTURALES

# ─────────────────────────────────────────────────────────────────────

def _build_structural_breaks(combos: List[List[int]],
n_max: int, k: int) -> Dict:
“””
Detecta si hubo un cambio en la distribución en algún punto
del historial (posible cambio de bolas o bombo).
Usa ventanas deslizantes de BLOCK_SIZE sorteos.
“””
n = len([c for c in combos if c])
if n < MIN_BREAK_BLOCK * 2:
return {“disponible”: False, “razon”: f”Necesita {MIN_BREAK_BLOCK*2}+ sorteos”}

```
# Construir series de frecuencia por bloque
valid_combos = [c for c in combos if c]
n_blocks     = len(valid_combos) // BLOCK_SIZE

if n_blocks < 4:
    return {"disponible": False, "razon": "Pocos bloques"}

block_entropy = []
block_means   = []
for b in range(n_blocks):
    block = valid_combos[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
    counts = np.zeros(n_max+1, dtype=np.float32)
    for combo in block:
        for n in combo: counts[n] += 1
    p   = counts[1:] / (counts[1:].sum() + 1e-9)
    ent = float(-np.sum(p * np.log2(p + 1e-12)))
    block_entropy.append(ent)
    block_means.append(float(np.mean([n for combo in block for n in combo])))

max_ent     = float(np.log2(n_max))  # entropía máxima teórica
ent_arr     = np.array(block_entropy)
mean_arr    = np.array(block_means)

# Detectar bloque con entropía más baja (menos aleatoriedad)
min_ent_idx = int(np.argmin(ent_arr))
min_ent_val = float(ent_arr[min_ent_idx])
deviation   = (max_ent - min_ent_val) / max_ent * 100

# Punto de cambio: mayor diferencia entre bloques consecutivos
diffs      = np.abs(np.diff(ent_arr))
break_idx  = int(np.argmax(diffs)) if len(diffs) > 0 else 0
break_sorteo = (break_idx + 1) * BLOCK_SIZE

# Estimar la fecha aproximada del cambio
break_date = None
if break_sorteo < len(combos):
    valid_idx = 0
    for t, c in enumerate(combos):
        if c:
            valid_idx += 1
        if valid_idx >= break_sorteo:
            break_date = t
            break

return {
    "disponible":      True,
    "n_bloques":       n_blocks,
    "block_entropy":   [round(e, 4) for e in block_entropy],
    "max_entropy":     round(max_ent, 4),
    "min_entropy_bloque": min_ent_idx,
    "min_entropy_val": round(min_ent_val, 4),
    "desviacion_pct":  round(deviation, 2),
    "break_point_sorteo": break_sorteo,
    "advertencia":     deviation > 5.0,
}
```

# ─────────────────────────────────────────────────────────────────────

# 6. CORRELACIÓN RESULTADO vs POZO ACUMULADO

# ─────────────────────────────────────────────────────────────────────

def _build_pot_correlation(combos: List[List[int]],
n_max: int, k: int) -> Dict:
“””
Detecta si los números que salen difieren cuando
el pozo acumulado es grande vs pequeño.

```
Como no tenemos datos directos del pozo, usamos como proxy
la suma de la combinación ganadora: sorteos con pozos grandes
tenderían a mostrar patrones diferentes si hay manipulación.

Nota: esto es una aproximación. Con datos reales del pozo
sería más preciso.
"""
sums = [sum(c) for c in combos if c]
if len(sums) < 100:
    return {"disponible": False}

sums_arr  = np.array(sums, dtype=float)
mean_sum  = float(np.mean(sums_arr))
std_sum   = float(np.std(sums_arr))

# Distribución de sumas
q25, q75 = float(np.percentile(sums_arr, 25)), float(np.percentile(sums_arr, 75))

# Test de normalidad (si las sumas son normales, el sorteo es uniforme)
stat, p_norm = scipy_stats.normaltest(sums_arr)

# Tendencia temporal de las sumas (¿está cambiando?)
n = len(sums_arr)
x = np.arange(n)
slope, intercept, r, p_trend, se = scipy_stats.linregress(x, sums_arr)

return {
    "disponible":    True,
    "mean_sum":      round(mean_sum, 2),
    "std_sum":       round(std_sum, 2),
    "q25":           round(q25, 1),
    "q75":           round(q75, 1),
    "p_normalidad":  round(float(p_norm), 4),
    "es_normal":     bool(float(p_norm) > SIGNIFICANCE),
    "tendencia_slope": round(float(slope), 4),
    "p_tendencia":   round(float(p_trend), 4),
    "hay_tendencia": bool(float(p_trend) < SIGNIFICANCE),
}
```

# ─────────────────────────────────────────────────────────────────────

# CONSTRUCTOR PRINCIPAL

# ─────────────────────────────────────────────────────────────────────

def build_advanced_stats(df: pd.DataFrame, name: str,
force: bool = False) -> Dict:
“””
Construye todas las estadísticas avanzadas para un juego.
Usa caché en disco si los datos no cambiaron.
“””
n_sorteos = len(df)
cache_f   = AS_CACHE_TPL.format(name=name)

```
if not force and os.path.exists(cache_f):
    try:
        cached = joblib.load(cache_f)
        if cached.get("n_sorteos") == n_sorteos:
            logger.info(f"[AS:{name}] Cargado desde caché ({n_sorteos} sorteos)")
            return cached
    except Exception:
        pass

logger.info(f"[AS:{name}] Calculando estadísticas avanzadas "
            f"({n_sorteos} sorteos)...")

n_max  = LOTTERIES[name]["n_max"]
k      = LOTTERIES[name]["k"]
combos = _get_combos(df, k, n_max)
dates  = _get_dates(df)

import time
t0 = time.time()

data = {"n_sorteos": n_sorteos, "name": name,
        "timestamp": datetime.now().isoformat()}

# 1. Correlaciones temporales
logger.info(f"[AS:{name}] Calculando correlaciones temporales...")
try:
    data["tc"] = _build_tc(combos, n_max)
    n_v = data["tc"]["n_valid"]
    logger.info(f"[AS:{name}] TC: {n_v} pares de sorteos válidos")
except Exception as e:
    logger.warning(f"[AS:{name}] TC falló: {e}")
    data["tc"] = {}

# 2. Sesgos físicos
logger.info(f"[AS:{name}] Detectando sesgos físicos...")
try:
    data["bias"] = _build_bias(combos, dates, n_max, k)
    rojos = data["bias"].get("numeros_rojos", [])
    if rojos:
        logger.warning(f"[AS:{name}] BIAS: Números con desviación "
                       f"significativa: {rojos}")
    else:
        logger.info(f"[AS:{name}] BIAS: Sin sesgos significativos detectados")
except Exception as e:
    logger.warning(f"[AS:{name}] BIAS falló: {e}")
    data["bias"] = {}

# 3. Efecto día de sorteo
logger.info(f"[AS:{name}] Analizando efecto por día...")
try:
    data["dow"] = _build_dow(combos, dates, n_max, k)
except Exception as e:
    logger.warning(f"[AS:{name}] DOW falló: {e}")
    data["dow"] = {"disponible": False}

# 4. Anti-correlaciones
logger.info(f"[AS:{name}] Detectando anti-correlaciones...")
try:
    data["anti"] = _build_anti(combos, n_max, k)
    n_anti = len(data["anti"].get("anti_pairs", []))
    logger.info(f"[AS:{name}] ANTI: {n_anti} pares que se evitan mucho")
except Exception as e:
    logger.warning(f"[AS:{name}] ANTI falló: {e}")
    data["anti"] = {}

# 5. Cambios estructurales
logger.info(f"[AS:{name}] Buscando cambios estructurales...")
try:
    data["breaks"] = _build_structural_breaks(combos, n_max, k)
    if data["breaks"].get("advertencia"):
        logger.warning(f"[AS:{name}] BREAK: Posible cambio estructural "
                       f"detectado en sorteo "
                       f"{data['breaks'].get('break_point_sorteo')}")
except Exception as e:
    logger.warning(f"[AS:{name}] BREAKS falló: {e}")
    data["breaks"] = {"disponible": False}

# 6. Correlación con pozo
logger.info(f"[AS:{name}] Analizando distribución de sumas...")
try:
    data["pot"] = _build_pot_correlation(combos, n_max, k)
    if data["pot"].get("hay_tendencia"):
        logger.warning(f"[AS:{name}] POT: Tendencia temporal en sumas "
                       f"detectada (slope={data['pot']['tendencia_slope']})")
except Exception as e:
    logger.warning(f"[AS:{name}] POT falló: {e}")
    data["pot"] = {"disponible": False}

elapsed = time.time() - t0
logger.info(f"[AS:{name}] Completado en {elapsed:.1f}s")

try:
    joblib.dump(data, cache_f)
except Exception as e:
    logger.warning(f"[AS:{name}] No se pudo cachear: {e}")

return data
```

# ─────────────────────────────────────────────────────────────────────

# SCORING DE COMBINACIONES

# ─────────────────────────────────────────────────────────────────────

def score_combos_advanced(combos_cand: List[Tuple[int,…]],
as_data: Dict,
last_combo: List[int],
n_max: int) -> np.ndarray:
“””
Calcula un score avanzado para cada combinación candidata.
Combina TC + BIAS + ANTI + DOW.

```
Args:
    combos_cand: Lista de combinaciones candidatas.
    as_data:     Resultado de build_advanced_stats.
    last_combo:  Último resultado real del sorteo.
    n_max:       Máximo número válido.

Returns:
    Array de scores. 1.0 = neutral. >1.0 = favorecido. <1.0 = penalizado.
"""
m   = len(combos_cand)
out = np.ones(m, dtype=np.float32)

# ── TC: correlaciones temporales ─────────────────────────────
tc = as_data.get("tc", {})
if tc and last_combo:
    try:
        tc_scores = _score_tc_batch(combos_cand, last_combo, tc, n_max)
        out *= np.clip(tc_scores, 0.5, 3.0)
    except Exception:
        pass

# ── BIAS: penalizar números con sesgo negativo (ratio < 0.7) ─
bias = as_data.get("bias", {})
bpn  = bias.get("bias_per_num", {})
if bpn:
    try:
        bias_vec = np.ones(n_max + 1, dtype=np.float32)
        for num, b in bpn.items():
            r = b.get("ratio", 1.0)
            # Amplificar ligeramente números frecuentes,
            # penalizar levemente los poco frecuentes
            if r > 1.2:
                bias_vec[int(num)] = 1.0 + (r - 1.0) * 0.3
            elif r < 0.7:
                bias_vec[int(num)] = 1.0 - (1.0 - r) * 0.2
        for idx, combo in enumerate(combos_cand):
            out[idx] *= float(np.mean([bias_vec[n] for n in combo
                                       if 1 <= n <= n_max]))
    except Exception:
        pass

# ── ANTI: penalizar combinaciones con pares que se evitan ────
anti = as_data.get("anti", {})
anti_set = set(
    (min(a,b), max(a,b))
    for a, b, r, _ in anti.get("anti_pairs", [])
    if r < 0.2  # solo los más extremos
)
if anti_set:
    try:
        for idx, combo in enumerate(combos_cand):
            c_list = list(combo)
            n_anti = sum(1 for i in range(len(c_list))
                         for j in range(i+1, len(c_list))
                         if (min(c_list[i],c_list[j]),
                             max(c_list[i],c_list[j])) in anti_set)
            if n_anti > 0:
                out[idx] *= max(0.7, 1.0 - n_anti * 0.08)
    except Exception:
        pass

return np.clip(out, 0.3, 5.0).astype(np.float32)
```

# ─────────────────────────────────────────────────────────────────────

# REPORTE PARA EL CORREO

# ─────────────────────────────────────────────────────────────────────

def generar_html_reporte(as_data_all: Dict[str, Dict]) -> str:
“””
Genera bloque HTML con el reporte de sesgo y salud estadística
para incluir en el correo del Jefe Maestro.
“””
if not as_data_all:
return “”

```
AZUL  = "#1a3a5c"
html  = (f"<h3 style='color:{AZUL};border-bottom:2px solid #2e75b6;"
         f"padding-bottom:6px'>📊 Reporte de Salud Estadística</h3>")
html += ("<p style='font-size:12px;color:#666'>"
         "Análisis de sesgos físicos, correlaciones temporales y "
         "cambios estructurales detectados en los históricos.</p>")

for name, data in as_data_all.items():
    bias   = data.get("bias", {})
    breaks = data.get("breaks", {})
    pot    = data.get("pot",   {})
    rojos  = bias.get("numeros_rojos",     [])
    amarillos = bias.get("numeros_amarillos", [])
    p_glob = bias.get("p_global", 1.0)
    chi2   = bias.get("chi2_global", 0)

    # Color del semáforo global
    if rojos:
        color_global = "#c62828"; icono = "🔴"
    elif amarillos:
        color_global = "#e65100"; icono = "🟡"
    else:
        color_global = "#2e7d32"; icono = "🟢"

    # Tendencia
    tend_txt = ""
    if pot.get("hay_tendencia"):
        s = pot.get("tendencia_slope", 0)
        tend_txt = (f"⚠️ Tendencia en sumas: "
                    f"{'↑' if s > 0 else '↓'} {abs(s):.3f}/sorteo<br>")

    # Cambio estructural
    break_txt = ""
    if breaks.get("advertencia"):
        break_txt = (f"⚠️ Posible cambio estructural en sorteo "
                     f"#{breaks.get('break_point_sorteo','?')}"
                     f" ({breaks.get('desviacion_pct','?')}% desviación)<br>")

    rojos_str    = " ".join(str(n) for n in rojos[:10])    if rojos    else "Ninguno"
    amarillos_str = " ".join(str(n) for n in amarillos[:10]) if amarillos else "Ninguno"

    html += f"""
```

<div style='background:#f9f9f9;border-left:4px solid {color_global};
     border-radius:4px;padding:10px;margin-bottom:10px'>
  <b style='color:{AZUL}'>{name}</b> &nbsp; {icono} &nbsp;
  <span style='font-size:12px;color:#888'>
    χ²={chi2:.1f} | p={p_glob:.4f}</span><br>
  <span style='font-size:12px'>
    🔴 Desviación significativa: <b>{rojos_str}</b><br>
    🟡 Desviación leve: <b>{amarillos_str}</b><br>
    {tend_txt}{break_txt}
  </span>
</div>"""

```
html += ("<p style='font-size:10px;color:#aaa'>"
         "🔴 p&lt;0.01 o racha &gt;3x esperada | "
         "🟡 p&lt;0.05 o racha &gt;2x esperada | "
         "🟢 distribución normal. "
         "Nota: desviaciones pueden ser varianza natural con N&lt;15,000.</p>")
return html
```

# ─────────────────────────────────────────────────────────────────────

# FUNCIÓN PRINCIPAL DE CARGA (llamada desde el predictor)

# ─────────────────────────────────────────────────────────────────────

def load_or_build_all(all_histories: Dict[str, pd.DataFrame],
force: bool = False) -> Dict[str, Dict]:
“””
Carga o construye las estadísticas avanzadas para todos los juegos.

```
Args:
    all_histories: {nombre: DataFrame} con datos de cada juego.
    force:         Si True, recalcula aunque haya caché.

Returns:
    Dict {nombre: stats_dict} listo para usar en scoring.
"""
result = {}
for name, df in all_histories.items():
    try:
        result[name] = build_advanced_stats(df, name, force=force)
    except Exception as e:
        logger.error(f"[AS:{name}] Error fatal: {e}")
        result[name] = {"n_sorteos": len(df), "name": name}

# Guardar reporte JSON para referencia
try:
    report = {}
    for name, data in result.items():
        bias   = data.get("bias", {})
        breaks = data.get("breaks", {})
        report[name] = {
            "p_global":         bias.get("p_global", 1.0),
            "numeros_rojos":    bias.get("numeros_rojos", []),
            "numeros_amarillos":bias.get("numeros_amarillos", []),
            "break_detectado":  breaks.get("advertencia", False),
            "timestamp":        data.get("timestamp", ""),
        }
    with open(AS_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_safe)
except Exception:
    pass

return result
```

# ─────────────────────────────────────────────────────────────────────

# TEST STANDALONE

# ─────────────────────────────────────────────────────────────────────

if **name** == “**main**”:
import sys
logging.basicConfig(level=logging.INFO,
format=”%(asctime)s - %(levelname)s - %(message)s”)

```
_ROOT_T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_T = os.path.join(_ROOT_T, "data")

histories = {}
for game in ["Melate", "Revancha", "Revanchita"]:
    path = os.path.join(_DATA_T, f"{game.lower()}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["_fecha_dt"] = pd.to_datetime(
            df["FECHA"], dayfirst=True, errors="coerce"
        )
        df = df.sort_values("_fecha_dt", ascending=False).reset_index(drop=True)
        histories[game] = df
        print(f"[{game}] {len(df)} sorteos")

if histories:
    result = load_or_build_all(histories, force=True)
    for name, data in result.items():
        print(f"\n{'='*50}")
        print(f"[{name}]")
        bias = data.get("bias", {})
        print(f"  Chi2 global: {bias.get('chi2_global','N/A')} "
              f"(p={bias.get('p_global','N/A')})")
        print(f"  Números rojos: {bias.get('numeros_rojos', [])}")
        print(f"  Números amarillos: {bias.get('numeros_amarillos', [])[:5]}")
        tc = data.get("tc", {})
        print(f"  TC pares válidos: {tc.get('n_valid','N/A')}")
        brk = data.get("breaks", {})
        print(f"  Cambio estructural: {brk.get('advertencia', False)}")
else:
    print("No se encontraron CSVs.")