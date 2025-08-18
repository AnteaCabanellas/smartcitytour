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
        return float(str(v).replace(",", "."))
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
# Carga y limpieza CSV
# ======================
try:
    # Forzar codificación y separador
    df = pd.read_csv("data/BBDDTUI_unida.csv", encoding="latin-1", sep=";")
    df.columns = df.columns.str.strip().str.upper()

    STR_COLS = [
        "NOMBRE_TUI", "DESCRIPCION_TUI", "CATEGORIA_TUI",
        "CATEGORIA_1", "CATEGORIA_2", "CATEGORIA_3",
        "DIRECCION_TUI", "WEBSITE", "TELEFONO", "HORARIO"
    ]
    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Saneado simple de web/teléfono
    if "WEBSITE" in df.columns:
        df["WEBSITE"] = df["WEBSITE"].where(df["WEBSITE"].str.contains(r"\.", na=False))
    if "TELEFONO" in df.columns:
        df["TELEFONO"] = df["TELEFONO"].where(df["TELEFONO"].str.replace(r"\D", "", regex=True).str.len() >= 7)

    # Coordenadas a float (soportando comas)
    for c in ["LATITUD_TUI", "LONGITUD_TUI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

except Exception as e:
    print("❌ Error al cargar CSV:", e)
    df = pd.DataFrame()

# Columnas relevantes para búsquedas
COLS_TXT = ["NOMBRE_TUI", "CATEGORIA_TUI", "CATEGORIA_1", "CATEGORIA_2", "CATEGORIA_3", "DESCRIPCION_TUI"]
present_cols = [c for c in COLS_TXT if c in df.columns]
if present_cols:
    df["_SEARCH_RAW"] = df[present_cols].fillna("").agg(" ".join, axis=1)
    df["_SEARCH"] = df["_SEARCH_RAW"].apply(norm_text)
    df["_TOKENS"] = df["_SEARCH"].apply(tokenize)
else:
    df["_SEARCH_RAW"] = ""
    df["_SEARCH"] = ""
    df["_TOKENS"] = [[] for _ in range(len(df))]

# ======================
# Parsing de horarios
# ======================
DOW_MAP = {
    "L": 0, "LUN": 0, "LUNES": 0,
    "M": 1, "MAR": 1, "MARTES": 1,
    "X": 2, "MIE": 2, "MIE": 2, "MIERCOLES": 2, "MIERCOLES": 2,
    "J": 3, "JUE": 3, "JUEVES": 3,
    "V": 4, "VIE": 4, "VIERNES": 4,
    "S": 5, "SAB": 5, "SABADO": 5, "SABADO": 5,
    "D": 6, "DOM": 6, "DOMINGO": 6
}
# Nota: duplicados para robustez sin acentos

def parse_horarios(s):
    """
    Devuelve dict {0..6: [(open_min, close_min), ...]} en minutos desde medianoche.
    Soporta patrones comunes: 'L-V 10:00-18:00; S 10:00-14:00'
    Si no hay parseo, devuelve vacío.
    """
    out = {i: [] for i in range(7)}
    if not s or not isinstance(s, str):
        return out
    ss = s.replace("–", "-").replace("—","-").replace(" a ", " ").replace("h", "").strip()
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
            frag = norm_text(frag).upper()
            if "-" in frag and len(frag) <= 5:
                a, b_ = frag.split("-")
                a_i, b_i = DOW_MAP.get(a[:3].upper(), None), DOW_MAP.get(b_[:3].upper(), None)
                if a_i is not None and b_i is not None:
                    if a_i <= b_i:
                        dias += list(range(a_i, b_i + 1))
                    else:
                        dias += list(range(a_i, 7)) + list(range(0, b_i + 1))
            else:
                i = DOW_MAP.get(frag[:3].upper(), None)
                if i is not None:
                    dias.append(i)

        def to_min(t):
            hh, mm = map(int, t.split(":"))
            return hh * 60 + mm

        o_m, c_m = to_min(o), to_min(c)
        for d in set(dias):
            out[d].append((o_m, c_m))
    return out

if "HORARIO" in df.columns:
    df["_CAL_HORAS"] = df["HORARIO"].apply(parse_horarios)
else:
    df["_CAL_HORAS"] = [{} for _ in range(len(df))]

# ======================
# Búsqueda tolerante (exacta → fuzzy)
# ======================
def buscar_top(pregunta, max_resultados=12):
    """Búsqueda tolerante: 1) coincidencias exactas por tokens normales, 2) fuzzy si hace falta."""
    if df.empty:
        return df.head(0)

    q = (pregunta or "").strip()
    qn = norm_text(q)
    q_tokens = tokenize(q)

    if not qn:
        return df.head(0)

    # --- FASE 1: exacta por tokens
    pattern = "|".join(map(re.escape, q_tokens)) if q_tokens else None

    def contains_any(s_norm):
        if not pattern:
            return False
        return re.search(pattern, s_norm) is not None

    name_norm = df.get("NOMBRE_TUI", pd.Series("", index=df.index)).apply(norm_text)
    cats_norm = df[[c for c in ["CATEGORIA_TUI", "CATEGORIA_1", "CATEGORIA_2", "CATEGORIA_3"] if c in df.columns]] \
                    .fillna("").agg(" ".join, axis=1).apply(norm_text)
    desc_norm = df.get("DESCRIPCION_TUI", pd.Series("", index=df.index)).apply(norm_text)

    score_exact = (
        name_norm.apply(contains_any).astype(int) * 5
        + cats_norm.apply(contains_any).astype(int) * 3
        + desc_norm.apply(contains_any).astype(int) * 1
    )

    hits = df.loc[score_exact > 0].copy()
    if len(hits) >= max_resultados:
        hits["__score"] = score_exact[score_exact > 0]
        return hits.sort_values(["__score", "NOMBRE_TUI"], ascending=[False, True]).head(max_resultados)

    # --- FASE 2: fuzzy
    def fuzzy_points(row):
        points = 0
        nm = str(row.get("NOMBRE_TUI", ""))
        if any(similar(nm, t) >= 0.72 for t in q_tokens):
            points += 5
        cats = " ".join(str(row.get(c, "")) for c in ["CATEGORIA_TUI", "CATEGORIA_1", "CATEGORIA_2", "CATEGORIA_3"])
        if any(similar(cats, t) >= 0.72 for t in q_tokens):
            points += 3
        desc = str(row.get("DESCRIPCION_TUI", ""))
        if any(similar(desc, t) >= 0.68 for t in q_tokens):
            points += 1
        return points

    fuzzy_scores = df.apply(fuzzy_points, axis=1)
    hits2 = df.loc[fuzzy_scores > 0].copy()

    if not hits2.empty:
        hits2["__score"] = fuzzy_scores[fuzzy_scores > 0] + score_exact.reindex(hits2.index, fill_value=0)
        if not hits.empty:
            comb = pd.concat([hits, hits2]).drop_duplicates()
            comb["__score"] = comb["__score"].fillna(0)
            return comb.sort_values(["__score", "NOMBRE_TUI"], ascending=[False, True]).head(max_resultados)
        else:
            return hits2.sort_values(["__score", "NOMBRE_TUI"], ascending=[False, True]).head(max_resultados)

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
        day_slots = slots.get(dow, [])
        if not day_slots:
            rows.append({**row.to_dict(), "__open_min": None, "__close_min": None, "__penalty": 1})
        else:
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

        # Elegir mejor índice
        best_idx = min(cand.index, key=lambda i: keyfun_row(cand.loc[i]))
        next_row = cand.loc[best_idx]

        open_m = next_row.get("__open_min")
        arrive = max(cur_min, open_m) if open_m is not None else cur_min
        dur = 60  # 1h por defecto
        leave = arrive + dur

        items.append((arrive, leave, next_row.to_dict()))
        cur_min = leave + 15  # buffer
        cur_lat = _to_float(next_row.get("LATITUD_TUI"))
        cur_lon = _to_float(next_row.get("LONGITUD_TUI"))
        used.add(best_idx)

    return items

def formatear_itinerario(items):
    def fmt(m):
        return f"{m//60:02d}:{m%60:02d}"
    agenda = []
    for arr, dep, row in items:
        agenda.append({
            "hora": f"{fmt(arr)}–{fmt(dep)}",
            "nombre": row.get("NOMBRE_TUI", "No disponible"),
            "direccion": row.get("DIRECCION_TUI", "No disponible"),
            "telefono": row.get("TELEFONO", "No disponible"),
            "web": row.get("WEBSITE", "No disponible"),
            "descripcion": row.get("DESCRIPCION_TUI", "No disponible")
        })
    return agenda

# ======================
# Detectar tipo de pregunta (tolerante)
# ======================
# ======================
# Detección de intención
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
    "alojamientos":[
        "alojamiento","hotel","hoteles","hostal","hostales","albergue","albergues",
        "apartahotel","apartahoteles","pensión","pensiones","casa de huéspedes",
        "camping","campings","residencia universitaria","residencias universitarias"
    ],
    "comida y bebida":[
        "restaurante","restaurantes","bar","bares","cafetería","cafeterias","café","cafes",
        "terraza","terrazas","coctelería","coctelerias","bar de copas","copas","chocolatería","chocolaterias","tapas"
    ],
    "eventos y vida nocturna":[
        "discoteca","discotecas","club","clubs","pub","pubs","karaoke","karaokes",
        "música en directo","concierto","conciertos","bingos","casino","casinos","bingos y casinos"
    ],
    "recreación y deporte":[
        "parque","parques","centro de ocio","centros de ocio","centro deportivo","centros deportivos",
        "instalaciones deportivas","gimnasio","gimnasios","piscina","piscinas","pista de hielo","pistas de hielo",
        "spa","spas","balneario","balnearios","golf","alquiler de bicicletas","bicicletas"
    ],
    "templos religiosos":[
        "iglesia","iglesias","mezquita","mezquitas","templo hindú","templos hindúes","templo hindu","templos hindues"
    ],
    "turismo":[
        "atracción turística","atracciones turísticas","atraccion turistica",
        "oficina de turismo","oficina turismo","guía turístico","guías turísticos",
        "guia turistico","guias turisticos","empresa de guías","empresas de guías",
        "parques y jardines","edificios y monumentos","consigna","espacios para eventos"
    ],
    "transporte":[
        "parada de bus","parada bus","autobús","bus","metro","estación de metro",
        "tren","estación de tren","estacion de tren","estacion de metro"
    ],
    "espacios culturales":[
        "museo","museos","galería","galerías","galeria","galerias",
        "biblioteca","bibliotecas","instalaciones culturales","zoológico","zoo","zoologico"
    ],
    "comercio":[ "centro comercial","centros comerciales","tienda","tiendas" ],
    "estudio":[ "escuela de cocina","escuelas de cocina","cata de vinos","catas de vinos","cata de aceites","catas de aceites","academia","taller","talleres" ],
    "oficinas y puntos de atención":[ "oficina","punto de atención","puntos de atención","atención al cliente" ],
}
KEYS_GENERAL = _dedupe_norm([w for ls in CATEGORY_SYNONYMS.values() for w in ls])
KEYS_PLAN = _dedupe_norm([
    "plan","planazo","planner","planificacion","planificación",
    "itinerario","ruta","tour","free tour","visita","visita guiada",
    "recorrido","agenda","programa","excursión","excursion",
    "hoy","mañana","tarde","noche","finde","fin de semana",
    "qué hacer","que hacer","donde ir","dónde ir"
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
    if hay_plan and hay_cat:
        return "planificacion"
    if hay_plan:
        return "planificacion"
    if hay_cat:
        return "general_con_datos"
    if similar(texto, "plan") >= 0.6 or similar(texto, "itinerario") >= 0.6:
        return "planificacion"
    return "general"

# ======================
# Resumen de info local (para respuestas generales)
# ======================
def resumen_para_respuesta(filas):
    if filas is None or filas.empty:
        return None
    partes = []
    for _, row in filas.iterrows():
        partes.append(
            f"\n🏛️ *{row.get('NOMBRE_TUI','Sin nombre')}* ({row.get('CATEGORIA_TUI','Sin categoría')})\n"
            f"📝 {row.get('DESCRIPCION_TUI','Sin descripción')}\n"
            + (f"📍 Dirección: {row['DIRECCION_TUI']}\n" if pd.notnull(row.get('DIRECCION_TUI')) else "")
            + (f"🕒 Horario: {row['HORARIO']}\n" if pd.notnull(row.get('HORARIO')) else "")
            + (f"📞 Teléfono: {row['TELEFONO']}\n" if pd.notnull(row.get('TELEFONO')) else "")
            + (f"🔗 Web: {row['WEBSITE']}\n" if pd.notnull(row.get('WEBSITE')) else "")
            + "--------------------------\n"
        )
    return "".join(partes).strip()

# ======================
# Rutas
# ======================
@app.route('/')
def index():
    # Si usas la plantilla mejorada, deja render_template('index.html')
    # Para pruebas rápidas, devolvemos un saludo:
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
        # 1) Buscar candidatos relevantes (tolerante)
        hits = buscar_top(user_latest, max_resultados=30)

        if hits.empty:
            # Fallback: toma una muestra variada (menos estricto)
            if df.empty:
                return jsonify({'response': "No tengo datos para armar un plan hoy."})
            random.seed(norm_text(user_latest))  # seed por consulta para consistencia
            sample_size = min(30, len(df))
            hits = df.sample(sample_size, random_state=random.randint(0, 10**6)).copy()

        # 2) Filtrar por si abren hoy (pero permitimos sin horario con penalización)
        cand = lugares_abiertos_hoy(hits)

        # 3) Construir plan determinista
        items = construir_itinerario(
            cand,
            start_time="09:30",
            start_lat=None,
            start_lon=None,
            max_stops=6
        )
        plan = formatear_itinerario(items)

        # 4) El modelo SOLO da formato (sin inventar)
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
            # Fallback textual simple si la API falla
            texto = ["Plan para hoy:"]
            for it in plan:
                texto.append(
                    f"- {it['hora']} · {it['nombre']} | {it['direccion']} | {it['telefono']} | {it['web']}"
                )
            return jsonify({'response': "\n".join(texto)})

    else:
        # Respuesta general usando datos locales
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
            # Fallback: devolver la lista tal cual
            return jsonify({'response': info_local})

# ======================
# Fine-tuning helpers
# ======================
def preparar_jsonl_desde_df(df_in, output_path="fine_tune_data.jsonl"):
    required_cols = ['NOMBRE_TUI', 'DESCRIPCION_TUI']
    if not all(col in df_in.columns for col in required_cols):
        raise ValueError(f"Columnas requeridas {required_cols} no están en el CSV.")

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df_in[required_cols].dropna().iterrows():
            record = {
                "messages": [
                    {"role": "system", "content": "Eres un asistente experto en turismo y actividades."},
                    {"role": "user", "content": f"Describe el sitio turístico llamado: {str(row['NOMBRE_TUI']).strip()}"},
                    {"role": "assistant", "content": str(row['DESCRIPCION_TUI']).strip()}
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
    
    # app.py
@app.route('/widget')
def widget():
    return render_template('chat_widget.html')  # el HTML del chat mejorado


# ======================
# Main
# ======================
if __name__ == "__main__":
    app.run(debug=True)


