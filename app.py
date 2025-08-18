# app.py
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
import json
import re
import unicodedata
import difflib
import random
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
import pytz
from openai import OpenAI
import time

# ======================
# Configuración básica
# ======================
load_dotenv()
app = Flask(__name__)

@app.context_processor
def inject_build_version():
    # Valor que forzará a refrescar el iframe del mapa cuando reinicies o se regenere el HTML
    return {'build_version': int(time.time())}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAD_TZ = pytz.timezone("Europe/Madrid")

# ======================
# Utilidades de texto / fuzzy
# ======================
def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("\x96", "-").replace("–", "-").replace("—", "-")  # <<< CAMBIO: normalizar guiones raros
    s = s.replace("?", "")  # <<< CAMBIO: limpiar marcas raras en horarios
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # quita acentos
    s = " ".join(s.split())  # colapsa espacios
    return s

def tokenize(s: str):
    s = norm_text(s)
    return [t for t in re.findall(r"[a-z0-9ñ]{2,}", s)]  # tokens de 2+ chars

def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()

# ======================
# Utilidades de números/geo
# ======================
def _to_float(v):
    try:
        return float(str(v).replace(",", "."))  # admite coma decimal
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(_to_float, [lat1, lon1, lat2, lon2])
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return 9999.0
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ======================
# Carga y limpieza CSV (adaptado a nuevas columnas)
# ======================
try:
    df = pd.read_csv("data/BBDDTUI_unida", encoding="latin-1", sep=";")
    df.columns = df.columns.str.strip().str.upper()

    # <<< CAMBIO: nuevas columnas principales y equivalencias
    COL_NOMBRE = "NOMBRE_TUI"
    COL_TIPOS = "TIPOS_TUI"
    COL_CATEG = "CATEGORIA_TUI"
    COL_DESC = "DESCRIPCION_TUI" if "DESCRIPCION_TUI" in df.columns else None  # podría no existir
    COL_DIR = "DIRECCION" if "DIRECCION" in df.columns else ("DIRECCION_TUI" if "DIRECCION_TUI" in df.columns else None)
    COL_URL = "URL" if "URL" in df.columns else "WEBSITE"
    COL_WEB = "WEBSITE" if "WEBSITE" in df.columns else "URL"
    COL_TEL = "TELEFONO" if "TELEFONO" in df.columns else None
    COL_HOR = "HORARIO" if "HORARIO" in df.columns else None
    COL_LAT = "LATITUD_TUI" if "LATITUD_TUI" in df.columns else None
    COL_LON = "LONGITUD_TUI" if "LONGITUD_TUI" in df.columns else None
    COL_RATING = "RATING_TUI" if "RATING_TUI" in df.columns else None

    STR_COLS = [c for c in [COL_NOMBRE, COL_TIPOS, COL_CATEG, COL_DESC, COL_DIR, COL_URL, COL_WEB, COL_TEL, COL_HOR] if c]
    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Limpieza web/teléfono
    if COL_WEB in df.columns:
        df[COL_WEB] = df[COL_WEB].where(df[COL_WEB].str.contains(r"\.", na=False))
    if COL_URL in df.columns:
        df[COL_URL] = df[COL_URL].where(df[COL_URL].str.contains(r"\.", na=False))
    if COL_TEL and COL_TEL in df.columns:
        df[COL_TEL] = df[COL_TEL].where(df[COL_TEL].str.replace(r"\D", "", regex=True).str.len() >= 7)

    # Coordenadas a float
    for c in [COL_LAT, COL_LON]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Rating a float (soporta coma)
    if COL_RATING and COL_RATING in df.columns:
        df["_RATING"] = pd.to_numeric(df[COL_RATING].astype(str).str.replace(",", ".", regex=False), errors="coerce")

except Exception as e:
    print("❌ Error al cargar CSV:", e)
    df = pd.DataFrame()

# ======================
# Columnas relevantes para búsquedas (adaptadas)
# ======================
def _split_csv_like(s: str):
    # Divide por coma y/o punto y coma para categorías/tipos combinados
    parts = [p.strip() for p in re.split(r"[;,]", s or "") if p.strip()]
    return parts

present_txt_cols = [c for c in [COL_NOMBRE, COL_CATEG, COL_TIPOS, COL_DESC] if c and c in df.columns]
if present_txt_cols:
    # Campos concatenados crudos
    df["_SEARCH_RAW"] = df[present_txt_cols].fillna("").agg(" ".join, axis=1)

    # Además, descomponemos CATEGORIA_TUI y TIPOS_TUI para mejorar matching por tokens
    def expand_cats_types(row):
        cats = _split_csv_like(row.get(COL_CATEG, "")) if COL_CATEG else []
        types = _split_csv_like(row.get(COL_TIPOS, "")) if COL_TIPOS else []
        return " ".join(cats + types)

    df["_SEARCH_AUX"] = df.apply(expand_cats_types, axis=1)
    df["_SEARCH"] = (df["_SEARCH_RAW"] + " " + df["_SEARCH_AUX"]).apply(norm_text)
    df["_TOKENS"] = df["_SEARCH"].apply(tokenize)
else:
    df["_SEARCH_RAW"] = ""
    df["_SEARCH"] = ""
    df["_TOKENS"] = [[] for _ in range(len(df))]

# ======================
# Parsing de horarios (ahora también inglés AM/PM)
# ======================
DOW_MAP_ES = {
    "L": 0, "LUN": 0, "LUNES": 0,
    "M": 1, "MAR": 1, "MARTES": 1,
    "X": 2, "MIE": 2, "MIERCOLES": 2,
    "J": 3, "JUE": 3, "JUEVES": 3,
    "V": 4, "VIE": 4, "VIERNES": 4,
    "S": 5, "SAB": 5, "SABADO": 5,
    "D": 6, "DOM": 6, "DOMINGO": 6
}
DOW_MAP_EN = {
    "MON": 0, "MONDAY": 0,
    "TUE": 1, "TUESDAY": 1,
    "WED": 2, "WEDNESDAY": 2,
    "THU": 3, "THURSDAY": 3,
    "FRI": 4, "FRIDAY": 4,
    "SAT": 5, "SATURDAY": 5,
    "SUN": 6, "SUNDAY": 6
}

def _to_min_24h(hhmm: str) -> int:
    hh, mm = map(int, hhmm.split(":"))
    return hh * 60 + mm

def _to_min_12h(hhmm_ampm: str) -> int:
    # acepta "9:00am", "09:30 PM", etc.
    s = norm_text(hhmm_ampm).replace(" ", "")
    m = re.match(r"^(\d{1,2}):(\d{2})(am|pm)$", s)
    if not m:
        # si viene sin am/pm, intentar 24h
        return _to_min_24h(hhmm_ampm)
    hh, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if hh == 12:
        hh = 0
    if ap == "pm":
        hh += 12
    return hh * 60 + mm

def parse_horarios_en(s: str):
    """
    Formatos tipo:
    'Monday: 9:00 AM–8:30 PM | Tuesday: 9:00 AM–8:30 PM | ... | Sunday: Closed'
    También soporta varias franjas separadas por coma.
    """
    out = {i: [] for i in range(7)}
    if not s or not isinstance(s, str):
        return out

    raw = s.replace("\x96", "-").replace("–", "-").replace("—", "-").replace("?", "")
    # separar por | o ;
    blocks = re.split(r"\s*\|\s*|;", raw)
    for b in blocks:
        if not b.strip():
            continue
        # "Monday: 12:00–14:30, 16:00–20:00"  (con o sin AM/PM)
        m = re.match(r"^\s*([A-Za-z]+)\s*:\s*(.+)$", b.strip())
        if not m:
            continue
        day_txt, times_txt = m.groups()
        day_key = norm_text(day_txt).upper()[:3]
        dow = DOW_MAP_EN.get(day_key)
        if dow is None:
            continue
        if "closed" in norm_text(times_txt):
            continue

        # separar franjas por coma
        spans = [x.strip() for x in times_txt.split(",") if x.strip()]
        for sp in spans:
            # detectar si viene am/pm
            has_ampm = bool(re.search(r"(?i)\b(am|pm)\b", sp))
            # "9:00 AM-8:30 PM" o "9:00-20:30"
            m2 = re.search(r"(\d{1,2}:\d{2})\s*(?:am|pm)?\s*-\s*(\d{1,2}:\d{2})\s*(?:am|pm)?", sp, re.IGNORECASE)
            if not m2:
                continue
            o, c = m2.group(1), m2.group(2)

            # Para obtener correctamente am/pm de cada extremo:
            # Extraemos sufijos reales si los hay
            suf = re.findall(r"(?i)(am|pm)", sp)
            if has_ampm:
                # heurística: si hay 2 sufijos, asignar; si hay 1, usarlo para el último y deducir el primero
                if len(suf) >= 2:
                    o_ap, c_ap = suf[0], suf[1]
                    o_m = _to_min_12h(f"{o}{o_ap}")
                    c_m = _to_min_12h(f"{c}{c_ap}")
                else:
                    c_ap = suf[-1]
                    c_m = _to_min_12h(f"{c}{c_ap}")
                    # deducir open: si hora cierre < 12pm y open >= 8 → probablemente am
                    # preferimos am por defecto
                    o_m = _to_min_12h(f"{o}am")
                    # si open >= close, probar pm
                    if o_m >= c_m:
                        o_m = _to_min_12h(f"{o}pm")
            else:
                o_m = _to_min_24h(o)
                c_m = _to_min_24h(c)

            out[dow].append((o_m, c_m))
    return out

def parse_horarios_es(s: str):
    """
    Español tipo 'L-V 10:00-18:00; S 10:00-14:00'
    """
    out = {i: [] for i in range(7)}
    if not s or not isinstance(s, str):
        return out
    ss = s.replace("\x96", "-").replace("–", "-").replace("—","-").replace(" a ", " ").replace("h", "").replace("?", "").strip()
    bloques = re.split(r"[;|/]+", ss)
    for b in bloques:
        b = b.strip()
        if not b:
            continue
        m = re.match(r"^([A-Za-zÁÉÍÓÚÑáéíóúñ\-\s,]+)\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", b)
        if not m:
            continue
        dias_txt, o, c = m.groups()
        dias = []
        for frag in re.split(r"[,\s]+", dias_txt.strip().upper()):
            if not frag:
                continue
            frag_n = norm_text(frag).upper()
            if "-" in frag_n and len(frag_n) <= 5:
                a, b_ = frag_n.split("-")
                a_i, b_i = DOW_MAP_ES.get(a[:3].upper(), None), DOW_MAP_ES.get(b_[:3].upper(), None)
                if a_i is not None and b_i is not None:
                    if a_i <= b_i:
                        dias += list(range(a_i, b_i + 1))
                    else:
                        dias += list(range(a_i, 7)) + list(range(0, b_i + 1))
            else:
                i = DOW_MAP_ES.get(frag_n[:3].upper(), None)
                if i is not None:
                    dias.append(i)
        o_m, c_m = _to_min_24h(o), _to_min_24h(c)
        for d in set(dias):
            out[d].append((o_m, c_m))
    return out

def parse_horarios(s: str):
    """
    Intenta primero EN (porque tu muestra viene en inglés), luego ES.
    """
    en = parse_horarios_en(s)
    if any(en[d] for d in en):
        return en
    return parse_horarios_es(s)

if "HORARIO" in df.columns:
    df["_CAL_HORAS"] = df["HORARIO"].apply(parse_horarios)
else:
    df["_CAL_HORAS"] = [{} for _ in range(len(df))]

# ======================
# Búsqueda tolerante (mejorada con categorías y tipos)
# ======================
def buscar_top(pregunta, max_resultados=12):
    if df.empty:
        return df.head(0)

    q = (pregunta or "").strip()
    qn = norm_text(q)
    q_tokens = tokenize(q)
    if not qn:
        return df.head(0)

    # Normalizaciones por columna
    name_norm = df.get(COL_NOMBRE, pd.Series("", index=df.index)).apply(norm_text)
    cat_norm = df.get(COL_CATEG, pd.Series("", index=df.index)).fillna("").apply(norm_text)
    tipos_norm = df.get(COL_TIPOS, pd.Series("", index=df.index)).fillna("").apply(norm_text)
    desc_norm = df.get(COL_DESC, pd.Series("", index=df.index)).fillna("").apply(norm_text) if COL_DESC else pd.Series("", index=df.index)

    pattern = "|".join(map(re.escape, q_tokens)) if q_tokens else None

    def contains_any(s_norm):
        if not pattern:
            return False
        return re.search(pattern, s_norm) is not None

    # ------- Fase 1: coincidencia "exacta" por tokens ponderada
    score_exact = (
        name_norm.apply(contains_any).astype(int) * 6
        + (cat_norm.apply(contains_any).astype(int) + tipos_norm.apply(contains_any).astype(int)) * 3
        + desc_norm.apply(contains_any).astype(int) * 1
    )

    hits = df.loc[score_exact > 0].copy()
    if len(hits) >= max_resultados:
        hits["__score"] = score_exact[score_exact > 0]
        # Asegura tipos numéricos para ordenación
        hits["__score"] = pd.to_numeric(hits["__score"], errors="coerce").fillna(0)
        return hits.sort_values(["__score", COL_NOMBRE], ascending=[False, True]).head(max_resultados)

    # ------- Fase 2: fuzzy
    def _split_csv_like(s: str):
        return [p.strip() for p in re.split(r"[;,]", s or "") if p.strip()]

    def fuzzy_points(row):
        points = 0
        nm = str(row.get(COL_NOMBRE, ""))
        if any(similar(nm, t) >= 0.75 for t in q_tokens):
            points += 6

        cats = " ".join(_split_csv_like(str(row.get(COL_CATEG, ""))))
        types = " ".join(_split_csv_like(str(row.get(COL_TIPOS, ""))))
        if any(similar(cats, t) >= 0.72 for t in q_tokens) or any(similar(types, t) >= 0.72 for t in q_tokens):
            points += 4

        desc = str(row.get(COL_DESC, "")) if COL_DESC else ""
        if desc and any(similar(desc, t) >= 0.68 for t in q_tokens):
            points += 2
        return points

    fuzzy_scores = df.apply(fuzzy_points, axis=1)
    hits2 = df.loc[fuzzy_scores > 0].copy()

    if not hits2.empty:
        # Combina con la puntuación exacta (si la hubo) para el mismo índice
        hits2["__score"] = (
            pd.to_numeric(fuzzy_scores[fuzzy_scores > 0], errors="coerce").fillna(0)
            + pd.to_numeric(score_exact.reindex(hits2.index, fill_value=0), errors="coerce").fillna(0)
        )

        if not hits.empty:
            # ✅ Deduplicación por índice (evita columnas no-hasheables)
            comb = pd.concat([hits, hits2], axis=0)
            comb = comb.loc[~comb.index.duplicated(keep="first")].copy()
            comb["__score"] = pd.to_numeric(comb.get("__score"), errors="coerce").fillna(0)
            return comb.sort_values(["__score", COL_NOMBRE], ascending=[False, True]).head(max_resultados)
        else:
            hits2["__score"] = pd.to_numeric(hits2.get("__score"), errors="coerce").fillna(0)
            return hits2.sort_values(["__score", COL_NOMBRE], ascending=[False, True]).head(max_resultados)

    return df.head(0)


# ======================
# Selección de lugares abiertos hoy e itinerario
# ======================
def lugares_abiertos_hoy(df_in, start_dt=None):
    now = start_dt or datetime.now(MAD_TZ)
    dow = now.weekday()  # 0=L ... 6=D
    rows = []
    idxs = []
    for idx, row in df_in.iterrows():
        slots = row["_CAL_HORAS"] if isinstance(row.get("_CAL_HORAS"), dict) else {}
        day_slots = slots.get(dow, []) if slots else []
        if not day_slots:
            rows.append({**row.to_dict(), "__open_min": None, "__close_min": None, "__penalty": 1})
        else:
            # coger la franja más temprana
            o_m, c_m = min(day_slots, key=lambda t: t[0])
            rows.append({**row.to_dict(), "__open_min": o_m, "__close_min": c_m, "__penalty": 0})
        idxs.append(idx)
    return pd.DataFrame(rows, index=idxs)

def construir_itinerario(df_hits, start_time="09:30", start_lat=None, start_lon=None, max_stops=6):
    def to_min(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    cur_min = to_min(start_time)
    items = []
    pool = df_hits.copy()
    used = set()
    cur_lat, cur_lon = _to_float(start_lat), _to_float(start_lon)

    for _ in range(max_stops):
        cand = pool.loc[~pool.index.isin(used)]
        if cand.empty:
            break

        def keyfun_row(row):
            pen = row.get("__penalty") or 0
            rlat = _to_float(row.get("LATITUD_TUI"))
            rlon = _to_float(row.get("LONGITUD_TUI"))
            dist = haversine_km(cur_lat, cur_lon, rlat, rlon) if cur_lat is not None and cur_lon is not None else 0
            oa = row.get("__open_min") if pd.notna(row.get("__open_min")) else 24 * 60
            return (pen, dist, oa, str(row.get("NOMBRE_TUI") or ""))

        best_idx = min(cand.index, key=lambda i: keyfun_row(cand.loc[i]))
        next_row = cand.loc[best_idx]

        open_m = next_row.get("__open_min")
        # 👇 fuerza int y maneja NaN/None
        if open_m is None or (isinstance(open_m, float) and pd.isna(open_m)):
            arrive = int(cur_min)
        else:
            arrive = int(max(cur_min, int(round(open_m))))

        dur = 60  # 1h por defecto
        leave = int(arrive + dur)

        items.append((arrive, leave, next_row.to_dict()))
        cur_min = int(leave + 15)  # buffer
        cur_lat = _to_float(next_row.get("LATITUD_TUI"))
        cur_lon = _to_float(next_row.get("LONGITUD_TUI"))
        used.add(best_idx)

    return items


def formatear_itinerario(items):
    def fmt(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return "??:??"
        m = int(round(m))  # 👈 asegura int
        return f"{m//60:02d}:{m%60:02d}"

    agenda = []
    for arr, dep, row in items:
        agenda.append({
            "hora": f"{fmt(arr)}–{fmt(dep)}",
            "nombre": row.get("NOMBRE_TUI", "No disponible"),
            "direccion": row.get("DIRECCION", row.get("DIRECCION_TUI", "No disponible")),
            "telefono": row.get("TELEFONO", "No disponible"),
            "web": row.get("WEBSITE", row.get("URL", "No disponible")),
            "descripcion": row.get("DESCRIPCION_TUI", "No disponible")
        })
    return agenda


# ======================
# Detección de intención (igual, pero robusta a tus nuevas categorías)
# ======================
def _dedupe_norm(words):
    seen = set()
    out = []
    for w in words:
        nw = norm_text(w)
        if nw and nw not in seen:
            seen.add(nw); out.append(nw)
    return out

CATEGORY_SYNONYMS = {
    # mantenemos tus grupos en español (encajan con 'Gastronomía y ocio nocturno', etc.)
    "alojamientos":[
        "alojamiento","hotel","hoteles","hostal","hostales","albergue","albergues",
        "apartahotel","apartahoteles","pension","pensiones","casa de huespedes",
        "camping","campings","residencia universitaria","residencias universitarias"
    ],
    "comida y bebida":[
        "restaurante","restaurantes","bar","bares","cafeteria","cafeterias","cafe","cafes",
        "terraza","terrazas","cocteleria","coctelerias","bar de copas","copas","chocolateria","chocolaterias","tapas",
        "gastronomia","ocio nocturno"
    ],
    "eventos y vida nocturna":[
        "discoteca","discotecas","club","clubs","pub","pubs","karaoke","karaokes",
        "musica en directo","concierto","conciertos","bingos","casino","casinos","bingos y casinos"
    ],
    "recreacion y deporte":[
        "parque","parques","centro de ocio","centros de ocio","centro deportivo","centros deportivos",
        "instalaciones deportivas","gimnasio","gimnasios","piscina","piscinas","pista de hielo","pistas de hielo",
        "spa","spas","balneario","balnearios","golf","alquiler de bicicletas","bicicletas","bienestar y deporte"
    ],
    "templos religiosos":[
        "iglesia","iglesias","mezquita","mezquitas","templo hindu","templos hindues","religion","basilica","catedral","monasterio","ermita","convento","templo budista"
    ],
    "turismo":[
        "atraccion turistica","atracciones turisticas",
        "oficina de turismo","guia turistico","guias turisticos",
        "parques y jardines","edificios y monumentos","consigna","espacios para eventos","naturaleza","senderismo","montanas","montaña"
    ],
    "transporte":[ "autobus","bus","metro","estacion de metro","tren","estacion de tren","aeropuerto","aparcamientos" ],
    "espacios culturales":[
        "museo","museos","galeria","galerias","biblioteca","bibliotecas","centro cultural","centros culturales","zoologico","zoologicos","cultura y arte","cines","teatros"
    ],
    "comercio":[ "centro comercial","centros comerciales","tienda","tiendas","compras","mercado","perfumeria","supermercado" ],
    "estudio":[ "escuela","universidad","colegio","academia","taller","talleres","libreria" ],
    "oficinas y puntos de atencion":[ "oficina","punto de atencion","atencion al cliente","ayuntamiento","hospital","clinica","consulado","embajada","instituciones y servicios publicos" ],
}
KEYS_GENERAL = _dedupe_norm([w for ls in CATEGORY_SYNONYMS.values() for w in ls])
KEYS_PLAN = _dedupe_norm([
    "plan","planazo","planner","planificacion","planificación",
    "itinerario","ruta","tour","free tour","visita","visita guiada",
    "recorrido","agenda","programa","excursion","excursión",
    "hoy","mañana","tarde","noche","finde","fin de semana",
    "que hacer","qué hacer","donde ir","dónde ir"
])

def contiene_fuzzy(texto, palabras, umbral=0.72):
    tks = tokenize(texto or "")
    for p in palabras:
        for t in tks:
            if similar(t, p) >= umbral:
                return True
    return False

def detectar_tipo_pregunta(texto):
    texto = texto or ""
    hay_plan = contiene_fuzzy(texto, KEYS_PLAN, umbral=0.7)
    hay_cat = contiene_fuzzy(texto, KEYS_GENERAL, umbral=0.7)
    if hay_plan:
        return "planificacion"
    if hay_cat:
        return "general_con_datos"
    if similar(texto, "plan") >= 0.6 or similar(texto, "itinerario") >= 0.6:
        return "planificacion"
    return "general"

# ======================
# Resumen de info local
# ======================
def resumen_para_respuesta(filas):
    if filas is None or filas.empty:
        return None
    partes = []
    for _, row in filas.iterrows():
        partes.append(
            f"\n🏛️ *{row.get('NOMBRE_TUI','Sin nombre')}* ({row.get('CATEGORIA_TUI','Sin categoría')})\n"
            f"📝 {row.get('DESCRIPCION_TUI','Sin descripción')}\n"
            + (f"📍 Dirección: {row.get('DIRECCION', row.get('DIRECCION_TUI'))}\n" if (row.get('DIRECCION') or row.get('DIRECCION_TUI')) else "")
            + (f"🕒 Horario: {row.get('HORARIO')}\n" if row.get('HORARIO') else "")
            + (f"📞 Teléfono: {row.get('TELEFONO')}\n" if row.get('TELEFONO') else "")
            + (f"🔗 Web: {row.get('WEBSITE', row.get('URL'))}\n" if (row.get('WEBSITE') or row.get('URL')) else "")
            + "--------------------------\n"
        )
    return "".join(partes).strip()

# ======================
# Rutas
# ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get("messages")
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400

    user_latest = [m['content'] for m in messages if m['role'] == 'user'][-1]
    tipo_pregunta = detectar_tipo_pregunta(user_latest)

    if tipo_pregunta == "planificacion":
        hits = buscar_top(user_latest, max_resultados=30)

        if hits.empty:
            if df.empty:
                return jsonify({'response': "No tengo datos para armar un plan hoy."})
            random.seed(norm_text(user_latest))
            sample_size = min(30, len(df))
            hits = df.sample(sample_size, random_state=random.randint(0, 10**6)).copy()

        cand = lugares_abiertos_hoy(hits)

        items = construir_itinerario(
            cand,
            start_time="09:30",
            start_lat=None,
            start_lon=None,
            max_stops=6
        )
        plan = formatear_itinerario(items)

        system_prompt = (
            "Eres un asistente experto en turismo.\n"
            "Fecha exacta: " + datetime.now(MAD_TZ).strftime("%d/%m/%Y") + " (Europe/Madrid).\n"
            "Recibirás un JSON con una agenda ya calculada.\n"
            "Tu tarea: presentarlo en español, organizado en Mañana / Mediodía / Tarde / Noche,\n"
            "sin añadir lugares ni modificar horarios. Si falta un dato, escribe 'No disponible'."
        )
        full_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Aquí está el plan (JSON):\n" + json.dumps(plan, ensure_ascii=False)}
        ]
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=full_messages,
                temperature=0.4,
                max_tokens=600,
            )
            reply = response.choices[0].message.content.strip()
            return jsonify({'response': reply})
        except Exception as e:
            print("🔥 ERROR formateando plan:", e)
            texto = ["Plan para hoy:"]
            for it in plan:
                texto.append(
                    f"- {it['hora']} · {it['nombre']} | {it['direccion']} | {it['telefono']} | {it['web']}"
                )
            return jsonify({'response': "\n".join(texto)})

    else:
        hits = buscar_top(user_latest, max_resultados=8)
        if hits.empty:
            sugerencias = (
                "- Ej: 'itinerario hoy museos'\n"
                "- 'bares con terraza'\n"
                "- 'piscinas públicas'\n"
                "- 'parques cercanos'\n"
                "- 'restaurantes familiares'"
            )
            reply = "No encontré coincidencias claras 🤔 (ahora soy menos estricto con faltas).\nPrueba con algo como:\n" + sugerencias
            return jsonify({'response': reply})

        info_local = resumen_para_respuesta(hits)
        if not info_local:
            reply = "Tengo datos, pero no pude formatearlos. Intenta acotar la búsqueda (p. ej., 'museos', 'bares')."
            return jsonify({'response': reply})

        system_prompt_content = (
            "Eres un asistente experto en turismo y actividades.\n"
            "Responde SOLO usando la información local dada a continuación.\n"
            "Incluye nombre, dirección, teléfono, URL y breve descripción cuando estén disponibles.\n"
            "No inventes datos y no añadas lugares que no estén en la lista."
        )

        full_messages = [
            {"role": "system", "content": system_prompt_content + "\n\n" + info_local},
            {"role": "user", "content": user_latest}
        ]

        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=full_messages,
                temperature=0.5,
                max_tokens=500,
            )
            reply = response.choices[0].message.content.strip()
            return jsonify({'response': reply})
        except Exception as e:
            print("🔥 ERROR general:", e)
            return jsonify({'response': info_local})

# ======================
# Fine-tuning helpers (sin cambios críticos)
# ======================
def preparar_jsonl_desde_df(df_in, output_path="fine_tune_data.jsonl"):
    required_cols = ['NOMBRE_TUI']
    if not all(col in df_in.columns for col in required_cols):
        raise ValueError(f"Falta NOMBRE_TUI en el CSV.")
    desc_col = 'DESCRIPCION_TUI' if 'DESCRIPCION_TUI' in df_in.columns else None

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df_in.iterrows():
            nombre = str(row['NOMBRE_TUI']).strip()
            desc = str(row.get(desc_col, "")).strip() if desc_col else ""
            if not nombre or not desc:
                continue
            record = {
                "messages": [
                    {"role": "system", "content": "Eres un asistente experto en turismo y actividades."},
                    {"role": "user", "content": f"Describe el sitio turístico llamado: {nombre}"},
                    {"role": "assistant", "content": desc}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path

@app.route('/fine-tune', methods=['POST'])
def lanzar_fine_tuning():
    try:
        jsonl_path = preparar_jsonl_desde_df(df)
        with open(jsonl_path, "rb") as f:
            upload_response = client.files.create(file=f, purpose="fine-tune")
        file_id = upload_response.id
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=os.getenv("FINE_TUNE_MODEL", "gpt-3.5-turbo")
        )
        job_id = fine_tune_response.id
        return jsonify({"status": "Fine-tune iniciado", "file_id": file_id, "job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/fine-tune/status/<job_id>', methods=['GET'])
def estado_fine_tuning(job_id):
    try:
        status_resp = client.fine_tuning.jobs.retrieve(job_id)
        return jsonify({
            "id": status_resp.id,
            "status": status_resp.status,
            "fine_tuned_model": getattr(status_resp, "fine_tuned_model", None),
            "created_at": status_resp.created_at,
            "updated_at": getattr(status_resp, "updated_at", None)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/widget')
def widget():
    return render_template('chat_widget.html')

# ======================
# Main
# ======================
if __name__ == "__main__":
    app.run(debug=True)
