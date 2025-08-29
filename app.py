# app.py
from flask import Flask, render_template, request, jsonify
import os, re, json, time, unicodedata, difflib, random
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, timedelta
import pandas as pd
import pytz
from dotenv import load_dotenv
from openai import OpenAI

# ======================
# Configuración básica
# ======================
load_dotenv()
app = Flask(__name__)

@app.context_processor
def inject_build_version():
    return {"build_version": int(time.time())}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAD_TZ = pytz.timezone("Europe/Madrid")
DAYS_ES = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

# ======================
# Utilidades de texto / fuzzy
# ======================
def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("\x96","-").replace("–","-").replace("—","-")
    s = s.replace("?","")
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s

# Tokenizador "neutro" (se usa para zonas/distritos para no romper señales como "en"/"por")
def tokenize(s: str):
    s = norm_text(s)
    return [t for t in re.findall(r"[a-z0-9ñ]+", s)]

# Stopwords para ranking (no se usan en extract_area_from_query)
STOPWORDS_ES = {
    "a","al","del","de","la","las","los","el","y","o","u","en","por","para","con","sin",
    "cerca","cercano","cercana","cercanos","cercanas","alrededor","sobre","entre","hacia","desde",
    "que","qué","donde","dónde","una","uno","un","unos","unas","mi","tu","su","me","te","se",
    "lo","le","les","nos","vos","ya","mas","más","esto","eso","aqui","aquí","alli","allí"
}
def tokenize_query(s: str):
    s = norm_text(s)
    toks = re.findall(r"[a-z0-9ñ]+", s)
    out = []
    for t in toks:
        if t in STOPWORDS_ES: 
            continue
        if len(t) < 3:       # KFC (3) sí pasa
            continue
        out.append(t)
    return out

def build_ngrams(tokens, nmin=1, nmax=4):
    grams = []
    for n in range(nmin, nmax+1):
        for i in range(len(tokens)-n+1):
            grams.append(" ".join(tokens[i:i+n]))
    return grams

def stem_es_light(t: str) -> str:
    t = norm_text(t)
    return re.sub(r"(as|os|a|o|es|s)$","", t)

def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()

# ======================
# Utilidades num/geo
# ======================
def _to_float(v):
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(_to_float, [lat1, lon1, lat2, lon2])
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ======================
# Limpieza de datos
# ======================
BOOL_TRUE = {"si","sí","true","1","y","yes"}
BOOL_FALSE = {"no","false","0","n"}

def clean_bool(val):
    s = norm_text("" if val is None else str(val))
    if s in BOOL_TRUE: return "SI"
    if s in BOOL_FALSE: return "NO"
    return ""

def clean_url(s):
    if not isinstance(s, str): return ""
    s = s.strip()
    if not s: return ""
    if s.startswith(("http://","https://")) or "." in s:
        return s
    return ""

def clean_phone(s):
    if not isinstance(s, str): s = str(s) if s is not None else ""
    digits = re.sub(r"\D+", "", s)
    return digits if len(digits) >= 7 else ""

def clean_decimal_comma(s):
    if s is None: return None
    ss = str(s).strip()
    if not ss or norm_text(ss) in {"nan","none","null"}: return None
    try:
        return float(ss.replace(",", "."))
    except Exception:
        return None

def clean_int(s):
    if s is None: return None
    ss = str(s).strip()
    if not ss or norm_text(ss) in {"nan","none","null"}: return None
    try:
        return int(float(ss))
    except Exception:
        return None

def _split_csv_like(s: str):
    return [p.strip() for p in re.split(r"[;,]", s or "") if p.strip()]

def _tipos_tokens(s: str):
    raw = (s or "").strip()
    if not raw: return []
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    return [norm_text(p) for p in parts]

def _tipo_head(s: str):
    raw = (s or "").strip()
    if not raw: return ""
    head = raw.split(",")[0].strip()
    return norm_text(head)

def _sanitize_name(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    s = re.sub(r"[�]+", "", s)           # caracteres inválidos
    s = re.sub(r"\?{2,}", "", s)         # secuencias de ?
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _looks_gibberish(name: str) -> bool:
    s = (name or "").strip()
    if len(s) < 3:         # KFC (3) permite, 1–2 letras no
        return True
    if re.fullmatch(r"[\W_¿?¡!.\-\"'()\s]+", s or ""):
        return True
    if s.count("?") >= max(3, len(s)//2):
        return True
    return False

# ======================
# Horarios
# ======================
DOW_MAP_ES = {"L":0,"LUN":0,"LUNES":0,"M":1,"MAR":1,"MARTES":1,"X":2,"MIE":2,"MIERCOLES":2,"J":3,"JUE":3,"JUEVES":3,"V":4,"VIE":4,"VIERNES":4,"S":5,"SAB":5,"SABADO":5,"D":6,"DOM":6,"DOMINGO":6}
DOW_MAP_EN = {"MON":0,"MONDAY":0,"TUE":1,"TUESDAY":1,"WED":2,"WEDNESDAY":2,"THU":3,"THURSDAY":3,"FRI":4,"FRIDAY":4,"SAT":5,"SATURDAY":5,"SUN":6,"SUNDAY":6}

def _to_min_24h(hhmm: str) -> int:
    hh, mm = map(int, hhmm.split(":"))
    hh = max(0, min(23, hh)); mm = max(0, min(59, mm))
    return hh*60 + mm

def _to_min_12h(hhmm_ampm: str) -> int:
    s = norm_text(hhmm_ampm).replace(" ", "")
    m = re.match(r"^(\d{1,2}):(\d{2})(am|pm)$", s)
    if not m: return _to_min_24h(hhmm_ampm)
    hh, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if hh == 12: hh = 0
    if ap == "pm": hh += 12
    return hh*60 + mm

def _preclean_horario(s: str) -> str:
    if not isinstance(s, str): return ""
    ss = s.strip()
    if norm_text(ss) in {"nan","none","null"}: return ""
    if ss.startswith("[") and ss.endswith("]"):
        parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", ss)
        parts = [p[0] or p[1] for p in parts]
        if parts: return " | ".join(parts)
    return ss

def parse_horarios_en(s: str):
    out = {i: [] for i in range(7)}
    if not s or not isinstance(s, str): return out
    s = _preclean_horario(s)
    raw = (s.replace("\x96","-").replace("–","-").replace("—","-").replace("?",""))

    if "open 24 hours" in norm_text(raw):
        for i in range(7): out[i].append((0, 24*60))
        return out

    blocks = re.split(r"\s*\|\s*|;", raw)
    for b in blocks:
        if not b.strip(): continue
        m = re.match(r"^\s*([A-Za-z]+)\s*:\s*(.+)$", b.strip())
        if not m: continue
        day_txt, times_txt = m.groups()
        dow = DOW_MAP_EN.get(norm_text(day_txt).upper()[:3])
        if dow is None: continue
        if "closed" in norm_text(times_txt): continue
        spans = [x.strip() for x in times_txt.split(",") if x.strip()]
        for sp in spans:
            has_ampm = bool(re.search(r"(?i)\b(am|pm)\b", sp))
            m2 = re.search(r"(\d{1,2}:\d{2})\s*(?:am|pm)?\s*-\s*(\d{1,2}:\d{2})\s*(?:am|pm)?", sp, re.IGNORECASE)
            if not m2: continue
            o, c = m2.group(1), m2.group(2)
            suf = re.findall(r"(?i)(am|pm)", sp)
            if has_ampm:
                if len(suf) >= 2:
                    o_m = _to_min_12h(f"{o}{suf[0].lower()}")
                    c_m = _to_min_12h(f"{c}{suf[1].lower()}")
                else:
                    c_m = _to_min_12h(f"{c}{suf[-1].lower()}")
                    o_m = _to_min_12h(f"{o}am")
                    if o_m >= c_m: o_m = _to_min_12h(f"{o}pm")
            else:
                o_m, c_m = _to_min_24h(o), _to_min_24h(c)
            if c_m <= o_m:
                out[dow].append((o_m, 24*60)); out[dow].append((0, c_m))
            else:
                out[dow].append((o_m, c_m))
    for d in range(7):
        slots = sorted(out[d], key=lambda t: (t[0], t[1]))
        merged = []
        for s0, s1 in slots:
            if not merged or s0 > merged[-1][1]:
                merged.append([s0, s1])
            else:
                merged[-1][1] = max(merged[-1][1], s1)
        out[d] = [(a, b) for a, b in merged]
    return out

def parse_horarios_es(s: str):
    out = {i: [] for i in range(7)}
    if not s or not isinstance(s, str): return out
    s = _preclean_horario(s)
    ss = (s.replace("\x96","-").replace("–","-").replace("—","-")
            .replace(" a "," ").replace("h","").replace("?","").strip())
    bloques = re.split(r"[;|/]+", ss)
    for b in bloques:
        b = b.strip()
        if not b: continue
        m = re.match(r"^([A-Za-zÁÉÍÓÚÑáéíóúñ\-\s,]+)\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", b)
        if not m: continue
        dias_txt, o, c = m.groups()
        dias = []
        for frag in re.split(r"[,\s]+", dias_txt.strip().upper()):
            if not frag: continue
            frag_n = norm_text(frag).upper()
            if "-" in frag_n and len(frag_n) <= 5:
                a, b_ = frag_n.split("-")
                a_i, b_i = DOW_MAP_ES.get(a[:3].upper(), None), DOW_MAP_ES.get(b_[:3].upper(), None)
                if a_i is not None and b_i is not None:
                    if a_i <= b_i: dias += list(range(a_i, b_i+1))
                    else: dias += list(range(a_i, 7)) + list(range(0, b_i+1))
            else:
                i = DOW_MAP_ES.get(frag_n[:3].upper(), None)
                if i is not None: dias.append(i)
        o_m, c_m = _to_min_24h(o), _to_min_24h(c)
        for d in set(dias):
            if c_m <= o_m:
                out[d].append((o_m, 24*60)); out[d].append((0, c_m))
            else:
                out[d].append((o_m, c_m))
    for d in range(7):
        slots = sorted(out[d], key=lambda t: (t[0], t[1]))
        merged = []
        for s0, s1 in slots:
            if not merged or s0 > merged[-1][1]:
                merged.append([s0, s1])
            else:
                merged[-1][1] = max(merged[-1][1], s1)
        out[d] = [(a, b) for a, b in merged]
    return out

def parse_horarios(s: str):
    en = parse_horarios_en(s)
    if any(en[d] for d in en): return en
    return parse_horarios_es(s)

def is_open_now(slots, now_dt=None):
    now = now_dt or datetime.now(MAD_TZ)
    dow = now.weekday()
    mins = now.hour*60 + now.minute
    if not isinstance(slots, dict): return False
    for o,c in slots.get(dow, []):
        if o <= mins <= c:
            return True
    return False

# ======================
# Carga y limpieza CSV
# ======================
CSV_PATH = "data/BBDD_TUI.csv"

def load_and_clean_df(csv_path=CSV_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=",")
    except Exception as e:
        print("❌ Error al cargar CSV:", e)
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.upper()

    COL_NOMBRE = "NOMBRE_TUI"
    COL_TIPOS  = "TIPOS_TUI"
    COL_CATEG  = "CATEGORIA_TUI"
    COL_DESC   = "DESCRIPCION_TUI" if "DESCRIPCION_TUI" in df.columns else None
    COL_DIR    = "DIRECCION" if "DIRECCION" in df.columns else ("DIRECCION_TUI" if "DIRECCION_TUI" in df.columns else None)
    COL_URL    = "URL" if "URL" in df.columns else "WEBSITE"
    COL_WEB    = "WEBSITE" if "WEBSITE" in df.columns else "URL"
    COL_TEL    = "TELEFONO" if "TELEFONO" in df.columns else None
    COL_HOR    = "HORARIO" if "HORARIO" in df.columns else None
    COL_LAT    = "LATITUD_TUI" if "LATITUD_TUI" in df.columns else None
    COL_LON    = "LONGITUD_TUI" if "LONGITUD_TUI" in df.columns else None
    COL_RATING = "RATING_TUI" if "RATING_TUI" in df.columns else None
    COL_REV    = "TOTAL_VALORACIONES_TUI" if "TOTAL_VALORACIONES_TUI" in df.columns else None
    COL_STATUS = "ESTADO_NEGOCIO" if "ESTADO_NEGOCIO" in df.columns else None
    COL_RES    = "RESERVA_POSIBLE" if "RESERVA_POSIBLE" in df.columns else None
    COL_ACC    = "ACCESIBILIDAD_SILLA_RUEDAS" if "ACCESIBILIDAD_SILLA_RUEDAS" in df.columns else None

    # strings básicos
    for c in [COL_NOMBRE, COL_TIPOS, COL_CATEG, COL_DESC, COL_DIR, COL_URL, COL_WEB, COL_TEL, COL_HOR, COL_STATUS, COL_RES, COL_ACC]:
        if c and c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].where(~df[c].str.match(r"(?i)^\s*(nan|none|null)\s*$"), "")

    # limpiar nombres y eliminar basura
    if COL_NOMBRE in df.columns:
        df[COL_NOMBRE] = df[COL_NOMBRE].apply(_sanitize_name)
        # descarta nombres claramente inválidos
        mask_bad = df[COL_NOMBRE].apply(_looks_gibberish)
        before = len(df)
        df = df.loc[~mask_bad].copy()
        removed = before - len(df)
        if removed:
            print(f"ℹ️ Filtradas {removed} filas con nombre inválido")

    # webs / urls
    if COL_WEB in df.columns:
        df[COL_WEB] = df[COL_WEB].apply(clean_url)
    if COL_URL in df.columns:
        df[COL_URL] = df[COL_URL].apply(clean_url)

    # telefono
    if COL_TEL and COL_TEL in df.columns:
        df[COL_TEL] = df[COL_TEL].apply(clean_phone)

    # coords
    for c in [COL_LAT, COL_LON]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # rating / reseñas
    if COL_RATING and COL_RATING in df.columns:
        rnum = df[COL_RATING].apply(clean_decimal_comma)
        rnum = rnum.where(rnum != 0.0, None)
        df["_RATING"] = pd.to_numeric(rnum, errors="coerce")
    if COL_REV and COL_REV in df.columns:
        df[COL_REV] = df[COL_REV].apply(clean_int)

    # estado negocio (descarta cerrados definitivos)
    if COL_STATUS and COL_STATUS in df.columns:
        st = df[COL_STATUS].astype(str).str.lower()
        mask_open = st.str.contains("abierto", na=True) & ~st.str.contains("cerrado permanentemente|cerrado definitivamente", na=False)
        df = df.loc[mask_open].copy()

    # booleans
    if COL_ACC and COL_ACC in df.columns:
        df[COL_ACC] = df[COL_ACC].apply(clean_bool).replace("", "NO")
    if COL_RES and COL_RES in df.columns:
        df[COL_RES] = df[COL_RES].apply(clean_bool).replace("", "NO")

    # horarios
    if COL_HOR and COL_HOR in df.columns:
        df["_CAL_HORAS"] = df[COL_HOR].apply(parse_horarios)
    else:
        df["_CAL_HORAS"] = [{} for _ in range(len(df))]

    # índice de búsqueda
    present_txt_cols = [c for c in [COL_NOMBRE, COL_TIPOS, COL_CATEG, COL_DESC] if c and c in df.columns]
    if present_txt_cols:
        df["_SEARCH_RAW"] = df[present_txt_cols].fillna("").agg(" ".join, axis=1)
        df["_TOK_TYPES"] = df.get(COL_TIPOS, pd.Series("", index=df.index)).apply(_tipos_tokens)
        df["_TOK_TYPE_HEAD"] = df.get(COL_TIPOS, pd.Series("", index=df.index)).apply(_tipo_head)
        df["_TOK_CATEG"] = df.get(COL_CATEG, pd.Series("", index=df.index)).fillna("").apply(lambda s: [norm_text(x) for x in _split_csv_like(s)])
        df["_SEARCH"] = (df["_SEARCH_RAW"]).apply(norm_text)
        df["_TOKENS"] = (df["_TOK_TYPES"] + df["_TOK_CATEG"])
    else:
        df["_SEARCH_RAW"] = ""; df["_SEARCH"] = ""; df["_TOKENS"] = [[] for _ in range(len(df))]

    # dedupe
    key_cols = []
    if "PLACE_ID" in df.columns: key_cols.append("PLACE_ID")
    if not key_cols:
        if "NOMBRE_TUI" in df.columns and ("DIRECCION" in df.columns or "DIRECCION_TUI" in df.columns):
            dir_col = "DIRECCION" if "DIRECCION" in df.columns else "DIRECCION_TUI"
            key_cols = ["NOMBRE_TUI", dir_col]
    if key_cols:
        df["_R_REV"] = pd.to_numeric(df.get("TOTAL_VALORACIONES_TUI"), errors="coerce").fillna(0).astype(int)
        df = (df
              .sort_values(by=["_RATING","_R_REV"], ascending=[False,False])
              .drop_duplicates(subset=key_cols, keep="first")
              .drop(columns=["_R_REV"], errors="ignore"))

    return df

df = load_and_clean_df(CSV_PATH)

# Vocabulario dinámico desde el CSV (sirve para cocinas/categorías sin listas duras)

TYPE_VOCAB = set()
if not df.empty:
    toks_col = df["_TOKENS"] if "_TOKENS" in df.columns else pd.Series([[]]*len(df), index=df.index)
    for toks in toks_col:
        if not isinstance(toks, (list, tuple)):
            toks = []  # garantizamos lista
        for t in toks:
            TYPE_VOCAB.add(stem_es_light(str(t)))

# ======================
# Sinónimos + áreas
# ======================
CABECERAS_TUI = {
    "hoteles","campings","restaurantes","bares","cafes","cafés","discotecas","museos","teatros",
    "iglesias","mezquitas","sinagogas","parques","zoologicos","zoológicos","parques de atracciones",
    "centros comerciales","tiendas","supermercados","gimnasios","spas","aeropuertos",
    "estaciones de tren","estaciones de metro","estaciones de autobus","estaciones de autobús",
    "ayuntamientos","hospitales","clinicas","clínicas","universidades","colegios",
    "campos de golf","acuarios","acuario"
}
HEAD_SYNONYMS = {
    "bar":"bares","vares":"bares","bares":"bares","pub":"bares","pubs":"bares","copas":"bares",
    "cafe":"cafés","café":"cafés","cafes":"cafés","cafés":"cafés",
    "restaurante":"restaurantes","restaurantes":"restaurantes",
    "museo":"museos","museos":"museos",
    "teatro":"teatros","teatros":"teatros",
    "discoteca":"discotecas","discotecas":"discotecas","club":"discotecas","clubs":"discotecas",
    "super":"supermercados","supermercado":"supermercados","supermercados":"supermercados",
    "gym":"gimnasios","gimnasio":"gimnasios","gimnasios":"gimnasios",
    "spa":"spas","spas":"spas",
    "parque":"parques","parques":"parques",
    "mall":"centros comerciales","centro comercial":"centros comerciales","centros comerciales":"centros comerciales",
    "tienda":"tiendas","tiendas":"tiendas",
}

DISTRICTS = {
    "centro","arganzuela","retiro","salamanca","chamartin","chamartín","tetuan","tetuán","chamberi","chamberí",
    "moncloa - aravaca","moncloa","aravaca","latina","carabanchel","usera","puente de vallecas","moratalaz",
    "ciudad lineal","hortaleza","villaverde","villa de vallecas","vicalvaro","vicálvaro",
    "san blas - canillejas","san blas","canillejas","barajas"
}
AREA_HINTS = {"por","en","cerca","cercanos","cercanas","zona","barrio","distrito","alrededor"}

def extract_area_from_query(q: str):
    tks = tokenize(q)  # OJO: sin stopwords
    joined, i = [], 0
    while i < len(tks):
        if i+2 <= len(tks) and f"{tks[i]} {tks[i+1]}" in {"san blas","ciudad lineal"}:
            joined.append(f"{tks[i]} {tks[i+1]}"); i += 2
        else:
            joined.append(tks[i]); i += 1
    for idx, t in enumerate(joined):
        if t in AREA_HINTS and idx+1 < len(joined):
            cand = joined[idx+1]
            if idx+2 < len(joined):
                two = f"{cand} {joined[idx+2]}"
                if two in DISTRICTS: return two
            if cand in DISTRICTS: return cand
    for t in joined:
        if t in DISTRICTS: return t
    return ""

def correct_head_typos(q_tokens):
    corrected = []
    all_heads = set(CABECERAS_TUI) | set(HEAD_SYNONYMS.keys()) | set(HEAD_SYNONYMS.values())
    for t in q_tokens:
        if t in HEAD_SYNONYMS:
            corrected.append(HEAD_SYNONYMS[t]); continue
        if t in all_heads:
            corrected.append(t); continue
        close = difflib.get_close_matches(t, list(all_heads), n=1, cutoff=0.8)
        corrected.append(HEAD_SYNONYMS.get(close[0], close[0]) if close else t)
    return corrected

# ======================
# Búsqueda base
# ======================
def buscar_top(pregunta, max_resultados=12):
    if df.empty: return df.head(0)

    q = (pregunta or "").strip()
    qn = norm_text(q)
    q_tokens_raw = tokenize_query(q)   # <-- con stopwords para ranking
    if not qn: return df.head(0)

    q_tokens = correct_head_typos(q_tokens_raw)
    q_ngrams = build_ngrams(q_tokens, 1, 4)
    area = extract_area_from_query(q)
    query_type_roots = {stem_es_light(t) for t in q_tokens if stem_es_light(t) in TYPE_VOCAB}

    COL_NOMBRE = "NOMBRE_TUI"
    COL_TIPOS  = "TIPOS_TUI"
    COL_CATEG  = "CATEGORIA_TUI"
    COL_DESC   = "DESCRIPCION_TUI" if "DESCRIPCION_TUI" in df.columns else None

    heads_in_query = {t for t in q_tokens if t in {norm_text(h) for h in CABECERAS_TUI} or t in set(HEAD_SYNONYMS.values())}
    base = df
    if heads_in_query:
        base = df.loc[df["_TOK_TYPE_HEAD"].isin(list(heads_in_query))].copy()
        if base.empty: base = df.copy()

    m_area = pd.Series([True]*len(base), index=base.index)
    if area:
        dir_col = "DIRECCION" if "DIRECCION" in base.columns else ("DIRECCION_TUI" if "DIRECCION_TUI" in base.columns else None)
        if dir_col:
            patt = re.escape(area)
            m_area = base[dir_col].astype(str)\
                .str.lower().str.normalize("NFKD")\
                .str.replace(r"[\u0300-\u036f]", "", regex=True)\
                .str.contains(patt, na=False)
    base = base.loc[m_area].copy()
    if base.empty: base = df.copy()

    name_norm  = base.get(COL_NOMBRE, pd.Series("", index=base.index)).apply(norm_text)
    tipos_norm = base.get(COL_TIPOS,  pd.Series("", index=base.index)).fillna("").apply(norm_text)
    cat_norm   = base.get(COL_CATEG,  pd.Series("", index=base.index)).fillna("").apply(norm_text)
    desc_norm  = base.get(COL_DESC,   pd.Series("", index=base.index)).fillna("").apply(norm_text) if COL_DESC else pd.Series("", index=base.index)
    tipo_head_series = base.get("_TOK_TYPE_HEAD", pd.Series("", index=base.index))

    pattern = "|".join(map(re.escape, q_tokens)) if q_tokens else None
    def contains_any(s_norm): return bool(pattern and re.search(pattern, s_norm))

    # boosts por n-gramas en el nombre (p.ej. "taco bell")
    def contains_phrase(series, phrase):
        patt = r"\b" + re.escape(phrase) + r"\b"
        return series.str.contains(patt, regex=True)

    phrase_boost = pd.Series(0, index=base.index, dtype="int64")
    for ph in q_ngrams:
        if len(ph) < 3: 
            continue
        phrase_boost = phrase_boost + contains_phrase(name_norm, ph).astype(int)

    # boost por tipos/categorías presentes (dinámico desde CSV)
    def row_has_any_root(row_tokens):
        rset = {stem_es_light(t) for t in (row_tokens or [])}
        return int(any(rt in rset for rt in query_type_roots))
    type_boost = base["_TOKENS"].apply(row_has_any_root) if query_type_roots else pd.Series(0, index=base.index, dtype="int64")

    score_exact = (
        name_norm.apply(contains_any).astype(int) * 6
        + tipo_head_series.apply(contains_any).astype(int) * 6
        + tipos_norm.apply(contains_any).astype(int) * 5
        + cat_norm.apply(contains_any).astype(int)   * 3
        + desc_norm.apply(contains_any).astype(int)  * 1
        + phrase_boost.astype(int) * 8               # << frases/marcas en nombre
        + type_boost.astype(int) * 5                 # << tipos/categorías (CSV)
    )

    if heads_in_query and ("teatros" in heads_in_query or "teatro" in heads_in_query):
        score_exact = (
            score_exact
            + tipos_norm.str.contains(r"\bteatro(\s+de\s+artes\s+escenicas)?\b", regex=True).astype(int) * 2
            - tipos_norm.str.contains(r"\b(cine|estudio|videobook|productora|films?)\b", regex=True).astype(int) * 2
        )

    hits = base.loc[score_exact > 0].copy()
    if len(hits) >= max_resultados:
        hits["__score"] = pd.to_numeric(score_exact[score_exact > 0], errors="coerce").fillna(0)
        return hits.sort_values(["__score","NOMBRE_TUI"], ascending=[False, True]).head(max_resultados)

    def fuzzy_points(row):
        points = 0
        nm = str(row.get("NOMBRE_TUI",""))
        if any(similar(nm, t) >= 0.75 for t in q_tokens): points += 6
        head = str(row.get("_TOK_TYPE_HEAD",""))
        if head and any(similar(head, t) >= 0.74 for t in q_tokens): points += 6
        tipos_join = " ".join(row.get("_TOK_TYPES", []))
        if tipos_join and any(similar(tipos_join, t) >= 0.72 for t in q_tokens): points += 5
        categ_join = " ".join(row.get("_TOK_CATEG", []))
        if categ_join and any(similar(categ_join, t) >= 0.72 for t in q_tokens): points += 3
        desc = str(row.get("DESCRIPCION_TUI","")) if "DESCRIPCION_TUI" in row else ""
        if desc and any(similar(desc, t) >= 0.68 for t in q_tokens): points += 2
        if "teatros" in heads_in_query:
            if re.search(r"\bteatro(\s+de\s+artes\s+esc[eé]nicas)?\b", tipos_join, flags=re.I): points += 4
            if re.search(r"\b(cine|estudio|videobook|productora|film(s)?)\b", tipos_join, flags=re.I): points -= 3
        return points

    fuzzy_scores = base.apply(fuzzy_points, axis=1)
    hits2 = base.loc[fuzzy_scores > 0].copy()

    if not hits2.empty:
        hits2["__score"] = pd.to_numeric(fuzzy_scores[fuzzy_scores > 0], errors="coerce").fillna(0) + \
                           pd.to_numeric(score_exact.reindex(hits2.index, fill_value=0), errors="coerce").fillna(0)
        if not hits.empty:
            comb = pd.concat([hits, hits2], axis=0)
            comb = comb.loc[~comb.index.duplicated(keep="first")].copy()
            comb["__score"] = pd.to_numeric(comb.get("__score"), errors="coerce").fillna(0)
            return comb.sort_values(["__score","NOMBRE_TUI"], ascending=[False, True]).head(max_resultados)
        else:
            hits2["__score"] = pd.to_numeric(hits2.get("__score"), errors="coerce").fillna(0)
            return hits2.sort_values(["__score","NOMBRE_TUI"], ascending=[False, True]).head(max_resultados)

    return base.head(0)

# ======================
# Filtros, distancia y listas
# ======================
def _aplica_filtros(hits, user_text):
    if hits is None or hits.empty: return hits
    tx = norm_text(user_text or "")
    out = hits.copy()

    if any(k in tx for k in ["abierto ahora","abiertos ahora","open now"]):
        out = out.loc[out["_CAL_HORAS"].apply(lambda s: is_open_now(s))]
    if any(k in tx for k in ["accesible","silla de ruedas","wheelchair"]):
        if "ACCESIBILIDAD_SILLA_RUEDAS" in out.columns:
            out = out.loc[out["ACCESIBILIDAD_SILLA_RUEDAS"].str.upper().eq("SI")]
    if any(k in tx for k in ["reserva","reservar","booking","book"]):
        if "RESERVA_POSIBLE" in out.columns:
            out = out.loc[out["RESERVA_POSIBLE"].str.upper().eq("SI")]

    return out if not out.empty else hits

def _sort_by_distance(df_in, user_lat, user_lon):
    if df_in is None or df_in.empty: return df_in
    if user_lat is None or user_lon is None: return df_in
    dists = []
    for _, r in df_in.iterrows():
        d = haversine_km(user_lat, user_lon, r.get("LATITUD_TUI"), r.get("LONGITUD_TUI"))
        dists.append(d if d is not None else 1e9)
    out = df_in.copy()
    out["_DIST_KM"] = dists
    by_cols = ["_DIST_KM"] + (["__score"] if "__score" in out.columns else [])
    by_asc  = [True] + ([False] if "__score" in out.columns else [])
    return out.sort_values(by_cols, ascending=by_asc)

def _fmt_bool_tick(v):
    s = str(v).strip().lower()
    if s in ["1","true","si","sí","yes","y"]: return "Sí"
    if s in ["0","false","no","n"]: return "No"
    return s if s else "No disponible"

def resumen_para_respuesta(filas):
    if filas is None or filas.empty: return None
    partes = []
    for i, (_, row) in enumerate(filas.iterrows(), start=1):
        rating = row.get("_RATING", None)
        rating_txt = f"{rating:.1f} ⭐" if pd.notna(rating) else "Sin rating"
        revs = row.get("TOTAL_VALORACIONES_TUI", "")
        try: revs_i = int(revs) if pd.notna(revs) else None
        except Exception: revs_i = None
        revs_txt = f" · {revs_i} reseñas" if revs_i is not None else ""
        acc = _fmt_bool_tick(row.get("ACCESIBILIDAD_SILLA_RUEDAS", ""))
        res = _fmt_bool_tick(row.get("RESERVA_POSIBLE", ""))
        web = row.get("WEBSITE","")
        url = row.get("URL","")
        web_line = f"🌐 Web: {web}\n" if web else ""
        map_line = f"🗺️ Mapa: {url}\n" if url and url != web else ""
        dist = row.get("_DIST_KM", None)
        dist_txt = f" · {dist:.1f} km" if (dist is not None and dist < 1e9) else ""

        partes.append(
            f"{i}. 🏛️ *{row.get('NOMBRE_TUI','(sin nombre)')}* ({row.get('TIPOS_TUI','Sin categoría')}){dist_txt}\n"
            + (f"📍 Dirección: {row.get('DIRECCION', row.get('DIRECCION_TUI'))}\n" if (row.get('DIRECCION') or row.get('DIRECCION_TUI')) else "")
            + (f"🕒 Horario: {row.get('HORARIO')}\n" if row.get('HORARIO') else "")
            + (f"📞 Teléfono: {row.get('TELEFONO')}\n" if row.get('TELEFONO') else "")
            + web_line + map_line
            + f"⭐ {rating_txt}{revs_txt}\n"
            + f"🦽 Accesible: {acc} · 📅 Reserva: {res}\n"
        )
    return "\n".join(partes).strip()

# ======================
# Planificación (día / semana)
# ======================
def lugares_abiertos_hoy(df_in, start_dt=None):
    now = start_dt or datetime.now(MAD_TZ)
    dow = now.weekday()
    rows, idxs = [], []
    for idx, row in df_in.iterrows():
        slots = row.get("_CAL_HORAS", {}) if isinstance(row.get("_CAL_HORAS"), dict) else {}
        day_slots = slots.get(dow, []) if slots else []
        if not day_slots:
            rows.append({**row.to_dict(), "__open_min": None, "__close_min": None, "__penalty": 1})
        else:
            o_m, c_m = min(day_slots, key=lambda t: t[0])
            rows.append({**row.to_dict(), "__open_min": o_m, "__close_min": c_m, "__penalty": 0})
        idxs.append(idx)
    return pd.DataFrame(rows, index=idxs)

def construir_itinerario(df_hits, start_time="09:30", start_lat=None, start_lon=None, max_stops=6, start_dt=None):
    def to_min(t): h, m = map(int, t.split(":")); return h*60 + m
    cur_min = to_min(start_time)
    items, used = [], set()
    pool = df_hits.copy()
    cur_lat, cur_lon = _to_float(start_lat), _to_float(start_lon)

    with_hours = pool["_CAL_HORAS"].apply(lambda s: bool(s and s.get((start_dt or datetime.now(MAD_TZ)).weekday())))
    few_hours = with_hours.sum() < max(2, int(0.4 * len(pool)))

    used_names = set()
    for _ in range(max_stops):
        cand = pool.loc[~pool.index.isin(used)]
        if cand.empty: break

        def keyfun_row(row):
            pen = row.get("__penalty") or 0
            rlat = _to_float(row.get("LATITUD_TUI")); rlon = _to_float(row.get("LONGITUD_TUI"))
            dist = haversine_km(cur_lat, cur_lon, rlat, rlon) if (cur_lat is not None and cur_lon is not None) else 0
            dist = dist if dist is not None else 1e6
            oa = row.get("__open_min") if pd.notna(row.get("__open_min")) else 24*60
            rating = float(row.get("_RATING") or 0.0)
            return (pen, dist, oa if not few_hours else -rating, -rating if not few_hours else oa, str(row.get("NOMBRE_TUI") or ""))

        best_idx = min(cand.index, key=lambda i: keyfun_row(cand.loc[i]))
        next_row = cand.loc[best_idx]

        nombre = str(next_row.get("NOMBRE_TUI","")).strip().lower()
        if nombre in used_names:
            used.add(best_idx); continue
        used_names.add(nombre)

        open_m = next_row.get("__open_min")
        arrive = int(max(cur_min, int(round(open_m))) if (open_m is not None and not pd.isna(open_m)) else cur_min)
        dur = 60
        leave = int(arrive + dur)

        items.append((arrive, leave, next_row.to_dict()))
        cur_min = int(leave + 15)
        cur_lat = _to_float(next_row.get("LATITUD_TUI")); cur_lon = _to_float(next_row.get("LONGITUD_TUI"))
        used.add(best_idx)

    return items

def formatear_itinerario(items):
    def fmt(m):
        if m is None or (isinstance(m, float) and pd.isna(m)): return "??:??"
        m = int(round(m)); return f"{m//60:02d}:{m%60:02d}"
    agenda = []
    seen = set()
    for arr, dep, row in items:
        nombre = row.get("NOMBRE_TUI","No disponible") or "No disponible"
        dire = row.get("DIRECCION", row.get("DIRECCION_TUI","No disponible")) or "No disponible"
        key = (str(nombre).strip().lower(), str(dire).strip().lower())
        if key in seen: continue
        seen.add(key)
        agenda.append({
            "hora": f"{fmt(arr)}–{fmt(dep)}",
            "nombre": nombre,
            "direccion": dire,
            "telefono": row.get("TELEFONO","No disponible"),
            "web": row.get("WEBSITE", row.get("URL","No disponible")),
            "mapa": row.get("URL",""),
            "descripcion": row.get("DESCRIPCION_TUI","No disponible")
        })
    return agenda

def _slot_bucket(hhmm):
    try:
        h = int(hhmm.split(":")[0])
    except Exception:
        return "Otros"
    if h < 12: return "Mañana"
    if h < 15: return "Mediodía"
    if h < 19: return "Tarde"
    return "Noche"

def render_plan_md_day(agenda_items, ciudad="Madrid"):
    buckets = {"Mañana":[], "Mediodía":[], "Tarde":[], "Noche":[]}
    for it in agenda_items:
        buckets[_slot_bucket(it["hora"][:5])].append(it)

    parts = [f"¡Claro! Aquí está tu agenda para hoy en {ciudad}:\n"]
    for sec in ["Mañana","Mediodía","Tarde","Noche"]:
        parts.append(f"🗓️ **{sec}:**")
        if not buckets[sec]:
            parts.append("• (sin propuestas)")
            continue
        for it in buckets[sec]:
            web = f" · 🌐 {it['web']}" if it.get("web") else ""
            mapa = f" · 🗺️ {it['mapa']}" if it.get("mapa") else ""
            parts.append(f"• {it['hora']} — *{it['nombre']}* ({it['direccion']}){web}{mapa}")
        parts.append("")
    return "\n".join(parts).strip()

def render_plan_md_week(days_blocks, ciudad="Madrid"):
    parts = [f"¡Listo! **Plan semanal en {ciudad}**:\n"]
    for d in days_blocks:
        parts.append(f"### 📅 {d['dia']} ({d['fecha']})")
        parts.append(render_plan_md_day(d["agenda"], ciudad))
        parts.append("")
    return "\n".join(parts).strip()

# ===== LLM helpers (fallback) =====
def _llm(messages, model=None, temperature=0.5, max_tokens=700):
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    resp = client.chat.completions.create(
        model=mdl, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def fallback_llm_plan(weekly: bool, start_time: str, user_query: str, lat=None, lon=None):
    coords = f"{lat},{lon}" if (lat is not None and lon is not None) else "desconocidas"
    sys = (
        "Eres un planificador turístico en Madrid. No tienes datos locales estructurados, "
        "así que genera un plan ESTIMATIVO y bonito con iconos. "
        "No inventes direcciones ni teléfonos. "
        "Usa lugares/barrios conocidos solo como referencia (sin datos exactos) y consejos prácticos."
    )
    if weekly:
        usr = (
            f"Quiero un plan SEMANAL en Madrid (horario de inicio sugerido {start_time}). "
            "Estructura Lunes–Domingo con bloques Mañana / Mediodía / Tarde / Noche. "
            "3–5 actividades por bloque, concisas, incluye tips (transporte, reservas, clima). "
            f"Mi consulta original fue: {user_query}. Coord. usuario: {coords}."
        )
    else:
        usr = (
            f"Quiero un plan para HOY en Madrid empezando a las {start_time}. "
            "Estructura por Mañana/Mediodía/Tarde/Noche, 3–5 actividades por bloque, "
            "consejos y alternativas si llueve. "
            f"Mi consulta original fue: {user_query}. Coord. usuario: {coords}."
        )
    return _llm([{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.5, max_tokens=900)

def construir_plan(user_latest, start_time, start_lat, start_lon, weekly=False, ciudad="Madrid"):
    hits = buscar_top(user_latest, max_resultados=60)
    if hits.empty:
        # Fallback total al LLM si el CSV no devuelve nada
        return fallback_llm_plan(weekly, start_time, user_latest, start_lat, start_lon)

    hits = _aplica_filtros(hits, user_latest)

    if weekly:
        base_date = datetime.now(MAD_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        blocks = []
        for off in range(7):
            dt = base_date + timedelta(days=off)
            cand = lugares_abiertos_hoy(hits, start_dt=dt)
            items = construir_itinerario(
                cand, start_time=start_time or "09:30",
                start_lat=start_lat, start_lon=start_lon,
                max_stops=6, start_dt=dt
            )
            agenda = formatear_itinerario(items)
            blocks.append({"dia": DAYS_ES[dt.weekday()], "fecha": dt.strftime("%d/%m"), "agenda": agenda})
        return render_plan_md_week(blocks, ciudad)

    # plan del día
    cand = lugares_abiertos_hoy(hits)
    items = construir_itinerario(
        cand, start_time=start_time or "09:30",
        start_lat=start_lat, start_lon=start_lon,
        max_stops=6
    )
    agenda = formatear_itinerario(items)
    return render_plan_md_day(agenda, ciudad)

# ======================
# Fallback al modelo (sin datos locales) – listados
# ======================
def fallback_llm_general(user_query: str, lat=None, lon=None) -> str:
    try:
        coords = f"{lat},{lon}" if (lat is not None and lon is not None) else "desconocidas"
        sys = (
            "Eres un asistente turístico útil. No tienes datos locales estructurados ahora mismo, "
            "así que responde de forma general y prudente, sin afirmar disponibilidad concreta de sitios. "
            "Sugiere categorías, zonas típicas, consejos de transporte y franjas horarias. "
            "No inventes direcciones ni teléfonos."
        )
        user = (
            f"Consulta: {user_query}\n"
            f"Coord. usuario: {coords}\n\n"
            "Devuelve 5–8 ideas en lista con iconos, breves y accionables."
        )
        return _llm([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.4, max_tokens=700)
    except Exception as e:
        print("🔥 ERROR fallback LLM:", e)
        return "No encontré datos locales y no pude consultar el modelo ahora mismo. Prueba a reformular la búsqueda."

# ======================
# Rutas
# ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages")
    if not messages:
        return jsonify({"error":"No messages provided"}), 400

    user_latest = [m["content"] for m in messages if m["role"] == "user"][-1]
    start_time = data.get("start_time") or "09:30"
    start_lat = data.get("start_lat")
    start_lon = data.get("start_lon")
    prefer_open = bool(data.get("prefer_open"))
    force_plan = bool(data.get("force_plan"))
    weekly = bool(data.get("weekly"))

    # 1) Planificación (día o semana)
    if force_plan or weekly:
        reply = construir_plan(
            user_latest,
            start_time=start_time,
            start_lat=start_lat,
            start_lon=start_lon,
            weekly=weekly,
            ciudad="Madrid"
        )
        return jsonify({"response": reply})

    # 2) Listado (sin plan): usar CSV; si vacío -> fallback LLM
    hits = buscar_top(user_latest, max_resultados=24)
    if prefer_open:
        user_latest = (user_latest + " abiertos ahora").strip()
    hits = _aplica_filtros(hits, user_latest)

    if start_lat is not None and start_lon is not None:
        hits = _sort_by_distance(hits, start_lat, start_lon)

    if hits.empty:
        reply = fallback_llm_general(user_latest, start_lat, start_lon)
        return jsonify({"response": reply})

    info_local = resumen_para_respuesta(hits.head(10))
    if not info_local:
        return jsonify({"response":"Tengo datos, pero no pude formatearlos. Intenta acotar la búsqueda (p. ej., 'museos', 'bares por chamberí')."})

    # Entregar directamente el listado (evitamos LLM si hay datos)
    return jsonify({"response": info_local})

# Auxiliares
@app.route("/widget")
def widget():
    return render_template("chat_widget.html")

@app.route("/health")
def health():
    return jsonify({"status":"ok", "rows": int(len(df))})

if __name__ == "__main__":
    app.run(debug=True)
