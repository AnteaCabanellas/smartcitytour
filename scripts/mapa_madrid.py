# -*- coding: utf-8 -*-
import shelve, time, json, re, unicodedata
import pandas as pd, folium
from tqdm import tqdm
from geopy.geocoders import Nominatim
from ratelimit import limits, sleep_and_retry
import os

# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sube desde scripts/ al raíz

CSV_FILE = os.path.join(ROOT, "data", "BBDD_TUI.csv")
HTML_OUT  = os.path.join(ROOT, "static", "mapa", "mapa_madrid.html")
CACHE_DB = os.path.join(ROOT, "data", "geocode_cache.db")

CAT2EMOJI = {
    "TRANSPORTE":                       "🚖",
    "GASOLINERAS Y APARCAMIENTOS":      "⛽",
    "ALOJAMIENTOS":                     "🏨",
    "ESPACIOS DEPORTIVOS":              "⚽",
    "OCIO NOCTURNO":                    "🍸",
    "ADMINISTRACIÓN PÚBLICA Y SEGURIDAD":"🏛️",
    "COMIDA Y BEBIDA":                  "🍽️",
    "ESPACIOS RELIGIOSOS":              "🛐",
    "TIENDAS DE COMIDA":                "🛒",
    "TIENDAS":                          "🛍️",
    "MODA Y BELLEZA":                   "💄",
    "EDUCACIÓN":                        "🎓",
    "SALUD Y MEDICINA":                 "🏥",
    "CULTURA, ARTE Y NATURALEZA":       "🎭",
    "OCIO":                              "🎡",
    "EVENTOS Y CELEBRACIONES":          "🎉",
    "SERVICIOS":                        "🛠️",
}


grupos = {
    "TRANSPORTE": [
        "Aeropuertos", "Estaciones De Autobús", "Estaciones De Metro", "Estaciones De Tren",
        "Estación De Transporte", "Heliopuerto", "Parada De Taxis", "Pista De Aterrizaje"
    ],
    "GASOLINERAS Y APARCAMIENTOS": [
        "Aparcamientos", "Gasolineras"
    ],
    "ALOJAMIENTOS": [
        "Alojamiento", "B&B (Cama Y Desayuno)", "Cabaña De Camping", "Cabañas", "Campings",
        "Casa De Huéspedes", "Complejo Residencial", "Edificio De Apartamentos",
        "Habitación De Huéspedes Privada", "Hostal", "Hotel De Larga Estancia",
        "Hotel Resort", "Hoteles", "Posada", "Área Para Autocaravanas", "Moteles"
    ],
    "ESPACIOS DEPORTIVOS": [
        "Campos De Golf", "Centro De Fitness", "Club Deportivo", "Coaching Deportivo",
        "Complejo Deportivo", "Estudio De Yoga", "Gimnasios", "Instalación Deportiva",
        "Piscinas", "Saunas", "Zona De Senderismo"
    ],
    "OCIO NOCTURNO": [
        "Discotecas", "Pub"
    ],
    "ADMINISTRACIÓN PÚBLICA Y SEGURIDAD": [
        "Ayuntamientos", "Oficina De Gobierno Local", "Oficina Gubernamental", "Policía"
    ],
    "COMIDA Y BEBIDA": [
        "Asador", "Bar De Vinos", "Bar Y Parrilla", "Bares", "Bocatería", "Cafetería Americana",
        "Cafés", "Café Para Perros", "Chocolatería", "Comida", "Comida Para Llevar",
        "Hamburguesería", "Heladerías", "Reparto De Comida", "Restaurante Africano",
        "Restaurante Asiático", "Restaurante Brasileño", "Restaurante Bufé", "Restaurante Chino",
        "Restaurante Coreano", "Restaurante De Alta Cocina", "Restaurante De Barbacoa",
        "Restaurante De Brunch", "Restaurante De Comida Rápida", "Restaurante De Desayunos",
        "Restaurante De Oriente Medio", "Restaurante De Ramen", "Restaurante De Sushi",
        "Restaurante Español", "Restaurante Estadounidense", "Restaurante Francés",
        "Restaurante Griego", "Restaurante Indio", "Restaurante Italiano", "Restaurante Japonés",
        "Restaurante Libanés", "Restaurante Mediterráneo", "Restaurante Mexicano",
        "Restaurante Tailandés", "Restaurante Turco", "Restaurante Vegano",
        "Restaurante Vegetariano", "Restaurante Vietnamita", "Restaurantes",
        "Panaderías", "Pastelerías", "Patio De Comidas", "Pizzería", "Marisquerías", "Salón De Té"
    ],
    "ESPACIOS RELIGIOSOS": [
        "Iglesias", "Lugar De Culto", "Mezquitas", "Sinagogas"
    ],
    "TIENDAS DE COMIDA": [
        "Carnicerías", "Charcuterías", "Mercados", "Mayorista", "Supermercado Asiático", "Supermercados"
    ],
    "TIENDAS": [
        "Tienda De Alimentación", "Tienda De Artículos Para El Hogar", "Tienda De Açaí",
        "Tienda De Bagels", "Tienda De Bicicletas", "Tienda De Bricolaje",
        "Tienda De Comestibles", "Tienda De Conveniencia", "Tienda De Deportes",
        "Tienda De Descuentos", "Tienda De Donuts", "Tienda De Dulces", "Tienda De Electrónica",
        "Tienda De Licores", "Tienda De Mascotas", "Tienda De Muebles", "Tienda De Móviles",
        "Tienda De Regalos", "Tienda De Repuestos", "Tienda De Ropa", "Tienda De Zumos",
        "Tiendas", "Zapaterías", "Gran Almacén", "Floristerías", "Confitería",
        "Ferretería", "Librerías"  
    ],
    "MODA Y BELLEZA": [
        "Barbería", "Cuidado Del Cabello", "Esteticista", "Estudio De Tatuajes Y Piercings",
        "Peluquerías", "Salón De Belleza", "Salón De Uñas", "Spas", "Sastre"
    ],
    "EDUCACIÓN": [
        "Bibliotecas", "Colegios", "Escuela Primaria", "Instituto", "Universidades", "Preescolar"
    ],
    "SALUD Y MEDICINA": [
        "Salud", "Clínica Dental", "Clínica Dermatológica", "Clínicas", "Dentistas", "Droguerías",
        "Farmacias", "Hospitales", "Fisioterapeuta", "Masajes", "Laboratorio Médico",
        "Médico", "Quiropráctico", "Centro De Bienestar", "Veterinaria"
    ],
    "CULTURA, ARTE Y NATURALEZA": [
        "Centro Cultural", "Lugar De Interés", "Lugar Emblemático Cultural", "Escultura", "Monumento",
        "Museos", "Plaza", "Sitio Histórico", "Teatro De Artes Escénicas", "Teatros",
        "Galería De Arte", "Estudio De Arte", "Acuarios", "Jardín", "Jardín Botánico",
        "Granjas", "Zoológicos"
    ],
    "OCIO": [
        "Cine", "Club De Comedia", "Centro De Ocio",
        "Boleras", "Montaña Rusa", "Parque Acuático", "Parque Estatal", "Parque Infantil",
        "Parque Para Perros", "Parques", "Parques De Atracciones", "Salón Recreativo",
        "Atracción Turística"
    ],
    "EVENTOS Y CELEBRACIONES": [
        "Sala De Baile", "Sala De Conciertos", "Sala De Eventos", "Salón De Bodas", "Recinto", "Servicio De Catering"
    ],
    "SERVICIOS": [
        "Cerrajero", "Electricista", "Fontanero", "Empresa De Mudanzas", "Empresa De Pintura",
        "Inmobiliarias", "Agencia De Cuidado Infantil", "Agencia De Excursiones", "Agencia De Viajes",
        "Almacenaje", "Aseguradora", "Consultoras", "Contratista General", "Mensajería",
        "Oficina Corporativa", "Operador De Telecomunicaciones", "Organizador De Campamentos De Verano",
        "Servicios Financieros", "Tanatorios", "Taller Mecánico", "Concesionario",
        "Abogado", "Campamento Infantil"
    ]
}

# Categoría de respaldo
if "OTROS" not in grupos:
    grupos["OTROS"] = []
if "OTROS" not in CAT2EMOJI:
    CAT2EMOJI["OTROS"] = "📌"

# ------------------------------------------------------------------
# CARGA CSV
df = pd.read_csv(CSV_FILE, sep=",", encoding="utf-8-sig", dtype=str, on_bad_lines="skip")

# ID único
df.reset_index(inplace=True)
df.rename(columns={"index": "ID"}, inplace=True)

def safe_get_series(col, default=""):
    if col in df.columns:
        s = df[col]
        return s if default is None else s.fillna(default)
    return pd.Series([default] * len(df), index=df.index)

df["DESCRIPCION_TUI"] = safe_get_series("DESCRIPCION_TUI", "").astype(str)
df["CATEGORIA_TUI"]   = safe_get_series("CATEGORIA_TUI", "").astype(str)
df["TIPOS_TUI"]       = safe_get_series("TIPOS_TUI", "").astype(str)

def safe_get(val, default=""):
    return default if (pd.isna(val) or (isinstance(val, str) and not val.strip())) else val

# Mostrar NaN/vacíos como "nan"
def nan_or_str(v):
    return "nan" if (pd.isna(v) or (isinstance(v, str) and not v.strip())) else str(v).strip()

# ------------------------------------------------------------------
# LAT/LON (arreglo coma decimal + extracción robusta)
lat_raw = safe_get_series("LATITUD_TUI", "").str.replace(",", ".", regex=False)
lon_raw = safe_get_series("LONGITUD_TUI", "").str.replace(",", ".", regex=False)
lat_num = pd.to_numeric(lat_raw.str.extract(r'(-?\d+(?:\.\d+)?)')[0], errors="coerce")
lon_num = pd.to_numeric(lon_raw.str.extract(r'(-?\d+(?:\.\d+)?)')[0], errors="coerce")
df["LAT"] = lat_num
df["LON"] = lon_num

# ------------------------------------------------------------------
# RATING_TUI y TOTAL_VALORACIONES_TUI (conversión a numérico)
rating_raw = safe_get_series("RATING_TUI", "")
# Soporta "4,2" -> 4.2 y extrae el número
rating_num = pd.to_numeric(
    rating_raw.str.replace(",", ".", regex=False).str.extract(r'(-?\d+(?:\.\d+)?)')[0],
    errors="coerce"
)
df["RATING"] = rating_num  # float

reviews_raw = safe_get_series("TOTAL_VALORACIONES_TUI", "")
reviews_num = pd.to_numeric(reviews_raw.str.extract(r'(\d+)')[0], errors="coerce")
df["TOTAL_REVIEWS"] = reviews_num  # int (NaN si falta)

# ------------------------------------------------------------------
# Geocoder (sólo si faltan coords)
geolocator = Nominatim(user_agent="mi_mapa_geocoder", timeout=10)

@sleep_and_retry
@limits(calls=1, period=1)
def geocode(addr: str):
    try:
        loc = geolocator.geocode(addr)
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass
    return float("nan"), float("nan")

cache = shelve.open(CACHE_DB)
for i, row in tqdm(df.iterrows(), total=len(df), desc="Geocodificando faltantes"):
    if pd.notna(row["LAT"]) and pd.notna(row["LON"]):
        continue
    addr = safe_get(row.get("DIRECCION_TUI", "")) or safe_get(row.get("DIRECCION", ""))
    if not addr:
        continue
    if addr in cache:
        lat, lon = cache[addr]
    else:
        lat, lon = geocode(addr)
        cache[addr] = (lat, lon)
        time.sleep(1.1)
    df.at[i, "LAT"] = lat
    df.at[i, "LON"] = lon
cache.close()

df = df.dropna(subset=["LAT", "LON"])
print("Filas válidas →", len(df))

# ------------------------------------------------------------------
# Normalización + utilidades
def norm(s: str) -> str:
    s = str(s or "").strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()

def split_normalize(cell: str):
    """Separa por , | / ; · normaliza para comparar · deduplica manteniendo orden."""
    parts = re.split(r"[,\|/;]+", str(cell or ""))
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        np = norm(p)
        if np not in seen:
            seen.add(np)
            out.append(p)
    return out

# Mapeos canónicos
norm_cat2canon = {norm(c): c for c in grupos.keys()}
tipo2cat = {}
valid_tipos_norm = set()
for cat, lista in grupos.items():
    for t in lista:
        nt = norm(t)
        valid_tipos_norm.add(nt)
        tipo2cat.setdefault(nt, set()).add(cat)

# ------------------------------------------------------------------
# Construcción de records con LÓGICA DE UNIÓN (OR)
records = []
present_types_by_cat = {cat: set() for cat in grupos}
cat_needing_empty = set()

for _, r in df.iterrows():
    # TIPOS reconocidos (canónicos)
    tipos_src = split_normalize(r.get("TIPOS_TUI", ""))
    tipos_valid = []
    for t in tipos_src:
        nt = norm(t)
        if nt in valid_tipos_norm:
            for cat, lista in grupos.items():
                for can in lista:
                    if norm(can) == nt:
                        tipos_valid.append(can)
                        break
                else:
                    continue
                break

    # CATEGORÍAS desde la columna
    cats_src = split_normalize(r.get("CATEGORIA_TUI", ""))
    cats_from_col = []
    for c in cats_src:
        nc = norm(c)
        if nc in norm_cat2canon:
            cats_from_col.append(norm_cat2canon[nc])

    # CATEGORÍAS derivadas desde TIPOS
    cats_from_types = []
    for t in tipos_valid:
        nt = norm(t)
        for c in tipo2cat.get(nt, []):
            if c not in cats_from_types:
                cats_from_types.append(c)

    # Unión OR (y fallback a OTROS)
    categories = []
    for c in cats_from_col + cats_from_types:
        if c not in categories:
            categories.append(c)
    if not categories:
        categories = ["OTROS"]

    subcats_display = list(dict.fromkeys(tipos_valid))

    cats_without_subs = set()
    for cat in categories:
        subs_in_cat = [s for s in subcats_display if s in grupos.get(cat, [])]
        if subs_in_cat:
            present_types_by_cat[cat].update(subs_in_cat)
        else:
            cats_without_subs.add(cat)
    if cats_without_subs:
        cat_needing_empty.update(cats_without_subs)

    val_acces = nan_or_str(r.get("ACCESIBILIDAD_SILLA_RUEDAS"))
    val_precio = nan_or_str(r.get("PRECIO"))
    val_estado = nan_or_str(r.get("ESTADO_NEGOCIO"))
    val_reserva = nan_or_str(r.get("RESERVA_POSIBLE"))

    # rating y total_reviews numéricos (None si NaN)
    rating_val = r.get("RATING")
    rating_out = float(rating_val) if pd.notna(rating_val) else None
    reviews_val = r.get("TOTAL_REVIEWS")
    reviews_out = int(reviews_val) if pd.notna(reviews_val) else None

    records.append({
        "id": int(r["ID"]),
        "name": safe_get(r.get("NOMBRE_TUI", "") or r.get("NOMBRE", "")),
        "description": safe_get(r.get("DESCRIPCION_TUI", "")),
        "lat": float(r["LAT"]),
        "lon": float(r["LON"]),
        "address": safe_get(r.get("DIRECCION_TUI", "") or r.get("DIRECCION", ""), f"{float(r['LAT']):.5f}, {float(r['LON']):.5f}"),
        "categories": categories,
        "subcategories": subcats_display,
        "email": safe_get(r.get("EMAIL", "")),
        "phone": safe_get(r.get("TELEFONO", "")),
        "website":   safe_get(r.get("CONTENT_URL", "") or r.get("WEBSITE", "")),
        "gmaps_url": safe_get(r.get("URL", "")),
        "horario": safe_get(r.get("HORARIO", "")),
        "precio": val_precio,
        "estado_negocio": val_estado,
        "reserva_posible": val_reserva,
        "accesibilidad_silla_ruedas": val_acces,
        # Nuevos campos
        "rating": rating_out,                 # float o null
        "total_reviews": reviews_out,         # int o null
    })

# cat2subs = subtipos realmente presentes por categoría (+ "(sin subcategoría)" donde haga falta)
cat2subs = {}
for cat in grupos:
    lst = sorted(present_types_by_cat.get(cat, set()), key=lambda s: s.casefold())
    if cat in cat_needing_empty and "(sin subcategoría)" not in lst:
        lst.append("(sin subcategoría)")
    if cat == "OTROS" and "(sin subcategoría)" not in lst:
        lst.append("(sin subcategoría)")
    cat2subs[cat] = lst

print("Registros totales:", len(df))
print("Con categorías por registro:", sum(1 for r in records if r["categories"]))
print("Con subtipos válidos:",      sum(1 for r in records if r["subcategories"]))

# ------------------------------------------------------------------
# Mapa
m = folium.Map(location=[40.4168, -3.7038], zoom_start=11, tiles="CartoDB positron")

def safe_json_dump_text(obj):
    txt = json.dumps(obj, ensure_ascii=False)
    return txt.replace("</script>", "<\\/script>")

def unique_values(field):
    return sorted({str(r[field]) for r in records if r.get(field) is not None})

unique_filters = {
    "acc_silla": unique_values("accesibilidad_silla_ruedas"),
    "precio":    unique_values("precio"),
    "estado":    unique_values("estado_negocio"),
    "reserva":   unique_values("reserva_posible"),
}

data_json           = safe_json_dump_text(records)
cat2subs_json       = safe_json_dump_text(cat2subs)
emoji_json          = safe_json_dump_text(CAT2EMOJI)
unique_filters_json = safe_json_dump_text(unique_filters)

# --- JS embebido ---
js = """
<style>
#filter-panel { position:relative; width:300px; max-height:85vh; font-family:sans-serif; z-index:5000; }
#filter-panel .panel { background:white; border-radius:8px; box-shadow:0 6px 20px rgba(0,0,0,0.25); padding:8px 12px 6px; display:flex; flex-direction:column; gap:6px; max-height:85vh; overflow:hidden; }
#filter-header { display:flex; justify-content:space-between; align-items:center; }
#filter-header .title-wrapper { display:flex; align-items:center; gap:6px; }
#filter-header h3 { margin:0; font-size:1.2em; font-weight:600; }
#panel-toggle { cursor:pointer; font-size:1.2em; user-select:none; }
#filter-body { display:flex; flex-direction:column; gap:6px; overflow:hidden; }
#filter-panel .buttons { display:flex; gap:6px; margin-top:2px; }
#filter-panel button { flex:1; padding:6px 10px; cursor:pointer; border:1px solid #d0d0d0; background:#f0f0f5; border-radius:5px; font-size:0.9em; transition:background .15s ease; }
#filter-panel button:hover { background:#e2e2ea; }
.category-block { border-top:1px solid #eee; padding-top:6px; margin-top:6px; }
.category-header { display:flex; align-items:center; cursor:pointer; gap:8px; user-select:none; padding:6px 8px; border-radius:6px; background:#fafafa; }
.category-header:hover { background:#f2f2f9; }
.category-header .title { flex:1; display:flex; align-items:center; gap:6px; }
.subcat-list { margin-left:4px; margin-top:4px; display:none; flex-direction:column; gap:6px; max-height:300px; overflow:auto; padding-left:10px; }
.toggle-arrow { width:16px; display:inline-block; transform:rotate(0deg); transition:transform .2s ease; font-weight:bold; }
.category-header.expanded .toggle-arrow { transform:rotate(90deg); }
.checkbox-wrapper { display:flex; align-items:center; gap:6px; padding:2px 0; }
.small-checkbox { width:16px; height:16px; }
.fav-heart { cursor:pointer; font-size:18px; user-select:none; }
.download-icon { cursor:pointer; font-size:18px; user-select:none; margin-left:auto; padding:2px 6px; border-radius:4px; transition:background .15s ease; }
.download-icon:hover { background:#e0e0ea; }
.leaflet-div-icon.emoji-pin{ background: transparent; border: none; font-size: 24px; line-height: 1; }
.leaflet-div-icon.user-pin { background: transparent; border: none; font-size: 28px; line-height: 1; }
</style>

<script id="data-json" type="application/json">""" + data_json + """</script>
<script id="cat2subs-json" type="application/json">""" + cat2subs_json + """</script>
<script id="emoji-json" type="application/json">""" + emoji_json + """</script>
<script id="filters-json" type="application/json">""" + unique_filters_json + """</script>

<script>
function getFoliumMap() {
  if (window._folium_map) return window._folium_map;
  for (const k in window) { try { if (window[k] instanceof L.Map) { window._folium_map = window[k]; return window[k]; } } catch {}
  }
  return null;
}
function withMap(cb, tries=0) {
  const m = getFoliumMap();
  if (m) cb(m); else if (tries < 60) setTimeout(()=>withMap(cb, tries+1), 100);
}

const rawData   = JSON.parse(document.getElementById('data-json').textContent   || "[]");
const cat2subs  = JSON.parse(document.getElementById('cat2subs-json').textContent || "{}");
const cat2emoji = JSON.parse(document.getElementById('emoji-json').textContent  || "{}");
const filtersUnique = JSON.parse(document.getElementById('filters-json').textContent || "{}");

const FAVORITES_KEY = "favorites_mapa";
let favorites = new Set(JSON.parse(localStorage.getItem(FAVORITES_KEY) || "[]"));
function saveFavorites(){ localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites])); }

let favoritesLayer = L.layerGroup();
const allMarkers = [];
const scheduleCache = new Map();

/* --- helpers de horario --- */
const DAY_INDEX = { "monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6,
  "lunes":0,"martes":1,"miércoles":2,"miercoles":2,"jueves":3,"viernes":4,"sábado":5,"sabado":5,"domingo":6 };
function normalizeSpaces(s){ return (s||"").replace(/[\\u2000-\\u200A\\u202F\\u205F\\u00A0]/g," ").trim(); }
function parseTimeToken(tok){ tok = normalizeSpaces(tok).replace(/\\./g,"").toUpperCase();
  const m = tok.match(/^\\s*(\\d{1,2})(?::(\\d{2}))?\\s*(AM|PM)?\\s*$/i);
  if(!m) return {hh:null, mm:null, mer:null, valid:false};
  let hh = parseInt(m[1],10), mm = m[2]?parseInt(m[2],10):0, mer = m[3]?m[3].toUpperCase():null;
  if(hh>24 || mm>59) return {hh:null, mm:null, mer:null, valid:false};
  return {hh, mm, mer, valid:true};
}
function toMinutes(hh,mm,mer){ if(mer==="AM"){ if(hh===12) hh=0; } else if(mer==="PM"){ if(hh!==12) hh+=12; } return hh*60+mm; }
function parseSchedule(horario){
  const sched = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]};
  if(!horario) return sched;
  const normalized = normalizeSpaces(horario).replace(/[—–−]/g,"-");
  const parts = normalized.split("|");
  for(const seg0 of parts){
    const seg = seg0.trim();
    const idxColon = seg.indexOf(":"); if(idxColon===-1) continue;
    const dayRaw = seg.slice(0, idxColon).trim().toLowerCase();
    const dayIdx = DAY_INDEX[dayRaw]; if(dayIdx===undefined) continue;
    let rest = seg.slice(idxColon+1).trim();
    if(/\\b(closed|cerrado)\\b/i.test(rest)){ sched[dayIdx]=[]; continue; }
    if(/(open\\s*24\\s*hours|abierto\\s*24\\s*horas|\\b24\\s*h\\b|\\b24\\s*horas\\b)/i.test(rest)){ sched[dayIdx]=[[0,1440]]; continue; }
    const intervals = rest.split(",").map(s=>s.trim()).filter(Boolean);
    for(const it of intervals){
      const m = it.split(/\\s*-\\s*/); if(m.length!==2) continue;
      const [a,b] = m; const t1=parseTimeToken(a), t2=parseTimeToken(b);
      if(!t1.valid || !t2.valid) continue;
      if(t1.mer==null && t2.mer!=null) t1.mer=t2.mer; if(t2.mer==null && t1.mer!=null) t2.mer=t1.mer;
      const m1=toMinutes(t1.hh,t1.mm,t1.mer), m2=toMinutes(t2.hh,t2.mm,t2.mer);
      if(Number.isNaN(m1)||Number.isNaN(m2)||m1===m2) continue;
      if(m1<m2) sched[dayIdx].push([m1,m2]); else { sched[dayIdx].push([m1,1440]); sched[(dayIdx+1)%7].push([0,m2]); }
    }
  }
  for(const d of Object.keys(sched)){
    const arr = sched[d].sort((x,y)=>x[0]-y[0]);
    const merged=[]; for(const [s,e] of arr){ if(!merged.length||s>merged[merged.length-1][1]) merged.push([s,e]); else merged[merged.length-1][1]=Math.max(merged[merged.length-1][1],e); }
    sched[d]=merged;
  }
  return sched;
}
function isOpenAtSchedule(sched, dow, minute){ return (sched[dow]||[]).some(([s,e])=>minute>=s && minute<e); }

/* --- util: sets --- */
function setHasAny(setA, arr){ for(const x of arr){ if(setA.has(x)) return true; } return false; }

/* --- Emoji dinámico --- */
function getEmojiForRec(catToSubs){
  const mains = Object.keys(catToSubs);
  for (const main of mains){
    const subs = catToSubs[main];
    for (const sub of subs){
      const chk = document.querySelector(`input[data-main='${main}'][data-sub='${sub}']`);
      if (chk && chk.checked){ return cat2emoji[main] || "📍"; }
    }
  }
  const fallback = mains[0];
  return cat2emoji[fallback] || "📍";
}

withMap((map)=>{
  // Panel de filtros
  const FilterControl = L.Control.extend({
    onAdd: function(){
      const container = L.DomUtil.create('div'); container.id='filter-panel'; container.className='leaflet-bar';
      container.innerHTML = `
        <div class="panel">
          <div id="filter-header">
            <div class="title-wrapper"><h3>Filtros</h3></div>
            <div id="panel-toggle">&#9660;</div>
          </div>
          <div id="filter-body">
            <div class="buttons">
              <button id="show-all">Mostrar todo</button>
              <button id="hide-all">Ocultar todo</button>
            </div>
            <div id="categories-container" style="overflow:auto; flex:1; margin-top:4px; padding-right:4px;"></div>
          </div>
        </div>`;
      L.DomEvent.disableClickPropagation(container);
      L.DomEvent.disableScrollPropagation(container);
      return container;
    },
    options: { position:'topright' }
  });
  map.addControl(new FilterControl());

  for (const rec of rawData) {
    const mains = (rec.categories && rec.categories.length) ? rec.categories : ["OTROS"];
    const allSubs = (rec.subcategories && rec.subcategories.length) ? rec.subcategories : [];

    const catToSubs = {};
    for (const main of mains){
      const validSubsForMain = (cat2subs[main]||[]);
      let subsForMain = allSubs.filter(s => validSubsForMain.includes(s));
      if (subsForMain.length === 0){
        if (validSubsForMain.includes("(sin subcategoría)")) subsForMain = ["(sin subcategoría)"];
      }
      if (subsForMain.length === 0 && main==="OTROS"){ subsForMain = ["(sin subcategoría)"]; }
      if (subsForMain.length > 0) catToSubs[main] = subsForMain;
    }

    let emoji = getEmojiForRec(catToSubs);

    const isFav = favorites.has(rec.id);
    const favSymbol = isFav ? "♥" : "♡";

    const websiteHTML = rec.website ? `<a href="${rec.website}" target="_blank" rel="noopener noreferrer">Abrir sitio</a>` : '—';
    const gmapsHTML   = rec.gmaps_url ? `<a href="${rec.gmaps_url}" target="_blank" rel="noopener noreferrer">Ver en Google Maps</a>` : '—';

    const catsHTML = mains.join(", ");
    const subsHTML = allSubs.length ? allSubs.join(", ") : "(sin subcategoría)";

    const ratingDisp = (rec.rating !== null && rec.rating !== undefined && !Number.isNaN(rec.rating)) ? rec.rating : '—';
    const reviewsDisp = (rec.total_reviews !== null && rec.total_reviews !== undefined && !Number.isNaN(rec.total_reviews)) ? rec.total_reviews : '—';

    const popupContent = `
      <div style="font-size:14px; max-width:850px; max-height:700px; overflow:auto">
        <b>${rec.name}</b>&nbsp;&nbsp;<span class="fav-heart" data-id="${rec.id}">${favSymbol}</span><br>
        <small>${rec.address}</small><br><br>
        ${rec.description || ""}<br><br>
        <strong>Categorías:</strong> ${catsHTML}<br>
        <strong>Tipos:</strong> ${subsHTML}<br>
        <strong>Sitio web:</strong> ${websiteHTML}<br>
        <strong>Google Maps:</strong> ${gmapsHTML}<br>
        <strong>Teléfono:</strong> ${rec.phone || '—'}<br>
        <strong>Precio:</strong> ${rec.precio}<br>
        <strong>Valorado:</strong> ${ratingDisp}<br>
        <strong>Número de reseñas:</strong> ${reviewsDisp}<br>
        <strong>Estado de negocio:</strong> ${rec.estado_negocio}<br>
        <strong>Reserva posible:</strong> ${rec.reserva_posible}<br>
        <strong>Accesibilidad Silla de ruedas:</strong> ${rec.accesibilidad_silla_ruedas}<br>
        <strong>Horario:</strong> ${rec.horario || '—'}<br>
      </div>`;

    const icon = L.divIcon({ className: "leaflet-div-icon emoji-pin", html: emoji, iconSize: [24,24], iconAnchor: [12,12] });
    const marker = L.marker([rec.lat, rec.lon], { icon }).bindPopup(popupContent, {maxWidth:900,maxHeight:800});

    marker._recId = rec.id;
    marker._rec   = rec;
    marker._catToSubs = catToSubs;
    marker._mainOrder = Object.keys(catToSubs);
    marker._currentEmoji = emoji;
    marker._updateEmoji = function(){
      const e = getEmojiForRec(marker._catToSubs);
      if (e !== marker._currentEmoji){
        marker._currentEmoji = e;
        marker.setIcon(L.divIcon({ className: "leaflet-div-icon emoji-pin", html: e, iconSize: [24,24], iconAnchor: [12,12] }));
      }
    };

    marker.addTo(map);
    allMarkers.push(marker);

    if (isFav) favoritesLayer.addLayer(marker);
  }

  favoritesLayer.addTo(map);

  map.on('popupopen', (e)=>{
    const pop = e.popup && e.popup.getElement(); if(!pop) return;
    const heart = pop.querySelector('.fav-heart');
    if (heart){
      const recId = parseInt(heart.getAttribute('data-id'));
      heart.addEventListener('click', ()=>{
        const isFav = favorites.has(recId);
        if (isFav){
          favorites.delete(recId); heart.textContent = "♡";
          favoritesLayer.eachLayer(l=>{ if(l._recId===recId) favoritesLayer.removeLayer(l); });
        } else {
          favorites.add(recId); heart.textContent = "♥";
          for (const mk of allMarkers){ if(mk._recId===recId) favoritesLayer.addLayer(mk); }
        }
        saveFavorites(); applyFilters();
      });
    }
  });

  function exportFavorites(){
    if(favorites.size===0) return alert("No has marcado favoritos todavía.");
    const headers = ["id","name","address","lat","lon","categories","subcategories","email","phone","website","gmaps_url","horario","description","rating","total_reviews"];
    const rows = [];
    for (const id of favorites) {
      const r = rawData.find(x=>x.id===id); if(!r) continue;
      const row = headers.map(h=> {
        let val = r[h];
        if(h==="categories") val=(r.categories||[]).join("|");
        if(h==="subcategories") val=(r.subcategories||[]).join("|");
        return `"` + String(val ?? "").replace(/"/g,'""') + `"`; }).join(";");
      rows.push(row);
    }
    const csv = [headers.join(";"), ...rows].join("\\r\\n");
    const blob = new Blob([csv], {type:"text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download="favoritos.csv";
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }

  const container = document.getElementById("categories-container");

  // FAVORITOS
  const favBlock = document.createElement("div"); favBlock.className="category-block";
  const favHeader = document.createElement("div"); favHeader.className="category-header";
  const favTitleW = document.createElement("div"); favTitleW.className="title";
  const favChk = document.createElement("input"); favChk.type="checkbox"; favChk.className="small-checkbox"; favChk.checked=true; favChk.id="fav-main-chk";
  const favEmoji = document.createElement("span"); favEmoji.textContent="❤";
  const favName = document.createElement("span"); favName.textContent="FAVORITOS";
  const downloadIcon = document.createElement("span"); downloadIcon.className="download-icon"; downloadIcon.textContent="📥"; downloadIcon.title="Descargar favoritos como CSV";
  favTitleW.appendChild(favChk); favTitleW.appendChild(favEmoji); favTitleW.appendChild(favName); favTitleW.appendChild(downloadIcon);
  favHeader.appendChild(favTitleW); favBlock.appendChild(favHeader); container.appendChild(favBlock);
  favChk.addEventListener('change', ()=>{ if(favChk.checked){ if(!map.hasLayer(favoritesLayer)) favoritesLayer.addTo(map); } else { if(map.hasLayer(favoritesLayer)) map.removeLayer(favoritesLayer); } });
  downloadIcon.addEventListener('click', (e)=>{ e.stopPropagation(); exportFavorites(); });

  // CATEGORÍAS
  const catMainBlock = document.createElement("div"); catMainBlock.className="category-block";
  const catMainHeader = document.createElement("div"); catMainHeader.className="category-header";
  const catMainTitleW = document.createElement("div"); catMainTitleW.className="title";
  const catMainChk = document.createElement("input"); catMainChk.type="checkbox"; catMainChk.className="small-checkbox"; catMainChk.checked=true; catMainChk.id="cat-main-chk";
  const catMainEmoji = document.createElement("span"); catMainEmoji.textContent="📁";
  const catMainName = document.createElement("span"); catMainName.textContent="CATEGORÍAS";
  catMainTitleW.appendChild(catMainChk); catMainTitleW.appendChild(catMainEmoji); catMainTitleW.appendChild(catMainName);
  const catMainArrow = document.createElement("div"); catMainArrow.className="toggle-arrow"; catMainArrow.innerHTML="&#9654;";
  catMainHeader.appendChild(catMainTitleW); catMainHeader.appendChild(catMainArrow);
  catMainBlock.appendChild(catMainHeader);
  const categoriesWrapper = document.createElement("div"); categoriesWrapper.className="subcat-list"; categoriesWrapper.style.paddingLeft="20px";

  for(const main of Object.keys(cat2subs)) {
    const block = document.createElement("div"); block.style.marginTop="8px";
    const header = document.createElement("div"); header.className="category-header"; header.style.padding="4px 6px";
    const titleWrapper = document.createElement("div"); titleWrapper.className="title";
    const chkCat = document.createElement("input"); chkCat.type="checkbox"; chkCat.className="small-checkbox"; chkCat.setAttribute("data-main", main); chkCat.checked=true;
    const emojiSpan = document.createElement("span"); emojiSpan.textContent = cat2emoji[main] || "📍";
    const nameSpan = document.createElement("span"); nameSpan.textContent = main;
    titleWrapper.appendChild(chkCat); titleWrapper.appendChild(emojiSpan); titleWrapper.appendChild(nameSpan);
    const arrow = document.createElement("div"); arrow.className="toggle-arrow"; arrow.innerHTML="&#9654;";
    header.appendChild(titleWrapper); header.appendChild(arrow); block.appendChild(header);
    const sublist = document.createElement("div"); sublist.className="subcat-list"; sublist.style.paddingLeft="20px";

    for (const sub of (cat2subs[main]||[])) {
      const row = document.createElement("div"); row.className="checkbox-wrapper";
      const chkSub = document.createElement("input"); chkSub.type="checkbox"; chkSub.checked=true; chkSub.setAttribute("data-main", main); chkSub.setAttribute("data-sub", sub);
      const lbl = document.createElement("span"); lbl.textContent=sub;
      row.appendChild(chkSub); row.appendChild(lbl); sublist.appendChild(row);
      chkSub.addEventListener("change", applyFilters);
    }
    block.appendChild(sublist); categoriesWrapper.appendChild(block);
    header.addEventListener("click", (e)=>{ e.stopPropagation(); const expanded=header.classList.toggle("expanded"); sublist.style.display = expanded ? "flex" : "none"; arrow.innerHTML = expanded ? "&#9660;" : "&#9654;"; });
    chkCat.addEventListener("change", ()=>{ const on=chkCat.checked; sublist.querySelectorAll("input[type=checkbox]").forEach(si=>{ si.checked=on; }); applyFilters(); });
  }
  catMainBlock.appendChild(categoriesWrapper); container.appendChild(catMainBlock);
  catMainHeader.classList.add("expanded"); categoriesWrapper.style.display="block"; catMainArrow.innerHTML="&#9660;";
  catMainHeader.addEventListener("click", ()=>{ const expanded=catMainHeader.classList.toggle("expanded"); categoriesWrapper.style.display = expanded ? "block" : "none"; catMainArrow.innerHTML = expanded ? "&#9660;" : "&#9654;"; });
  catMainChk.addEventListener("change", ()=>{ const on=catMainChk.checked; categoriesWrapper.querySelectorAll("input[type=checkbox]").forEach(chk=>{ chk.checked=on; }); applyFilters(); });

  // HORARIO (sin checkbox maestro, con botones Aplicar/Quitar)
  const hourBlock = document.createElement("div"); hourBlock.className="category-block";
  const hourHeader = document.createElement("div"); hourHeader.className="category-header";
  const hourTitleW = document.createElement("div"); hourTitleW.className="title";
  const hourEmoji = document.createElement("span"); hourEmoji.textContent="⏰";
  const hourName = document.createElement("span"); hourName.textContent="HORARIO";
  hourTitleW.appendChild(hourEmoji); hourTitleW.appendChild(hourName);
  const hourArrow = document.createElement("div"); hourArrow.className="toggle-arrow"; hourArrow.innerHTML="&#9654;";
  hourHeader.appendChild(hourTitleW); hourHeader.appendChild(hourArrow);
  hourBlock.appendChild(hourHeader);

  const hoursWrapper = document.createElement("div"); hoursWrapper.className="subcat-list"; hoursWrapper.style.paddingLeft="12px"; hoursWrapper.id="hours-wrapper";

  const dayRow = document.createElement("div"); dayRow.className="checkbox-wrapper"; dayRow.style.alignItems="center";
  const dayLbl = document.createElement("label"); dayLbl.textContent = "Día:"; dayLbl.style.minWidth = "40px";
  const daySelect = document.createElement("select"); daySelect.id="day-select"; daySelect.style.flex="1";
  ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"].forEach((d,i)=>{ const opt=document.createElement("option"); opt.value=String(i); opt.textContent=d; daySelect.appendChild(opt); });
  dayRow.appendChild(dayLbl); dayRow.appendChild(daySelect);

  const timeRow = document.createElement("div"); timeRow.className="checkbox-wrapper"; timeRow.style.alignItems="center";
  const timeLbl = document.createElement("label"); timeLbl.textContent = "Hora:"; timeLbl.style.minWidth = "40px";
  const timeInput = document.createElement("input"); timeInput.type="time"; timeInput.id="time-input"; timeInput.step=60;
  timeRow.appendChild(timeLbl); timeRow.appendChild(timeInput);

  const actionsRow = document.createElement("div"); actionsRow.style.display="flex"; actionsRow.style.gap="6px"; actionsRow.style.marginTop="6px";
  const applyBtn = document.createElement("button"); applyBtn.textContent="Aplicar horario";
  const clearBtn = document.createElement("button"); clearBtn.textContent="Quitar filtro";
  actionsRow.appendChild(applyBtn); actionsRow.appendChild(clearBtn);

  const hint = document.createElement("div"); hint.style.fontSize="12px"; hint.style.color="#666"; hint.textContent = "Elige día y hora y pulsa “Aplicar horario”.";
  hoursWrapper.appendChild(dayRow); hoursWrapper.appendChild(timeRow); hoursWrapper.appendChild(actionsRow); hoursWrapper.appendChild(hint);

  hourBlock.appendChild(hoursWrapper);
  container.appendChild(hourBlock);

  hourHeader.addEventListener("click", (e)=>{
    e.stopPropagation();
    const expanded=hourHeader.classList.toggle("expanded");
    hoursWrapper.style.display = expanded ? "block" : "none";
    hourArrow.innerHTML = expanded ? "&#9660;" : "&#9654;";
  });

  // valores por defecto (día y hora actuales)
  (function setDefaultDayTime(){
    const now=new Date();
    daySelect.value=String((now.getDay()+6)%7);
    const hh=String(now.getHours()).padStart(2,"0");
    const mm=String(now.getMinutes()).padStart(2,"0");
    timeInput.value = `${hh}:${mm}`;
  })();

  // estado del filtro de horario controlado por botones
  let hourFilterActive = false;

  applyBtn.addEventListener("click", ()=>{
    if(!timeInput.value) { alert("Selecciona una hora."); return; }
    hourFilterActive = true;
    applyFilters();
  });
  clearBtn.addEventListener("click", ()=>{
    hourFilterActive = false;
    applyFilters();
  });

  // === Helpers filtros simples (OR interno, AND externo) ===
  function createSimpleFilterBlock(opts){
    const block  = document.createElement("div"); block.className="category-block";
    const header = document.createElement("div"); header.className="category-header";
    const titleW = document.createElement("div"); titleW.className="title";
    const em = document.createElement("span"); em.textContent = opts.emoji || "🗂️";
    const nm = document.createElement("span"); nm.textContent = opts.title;
    titleW.appendChild(em); titleW.appendChild(nm);
    const arrow = document.createElement("div"); arrow.className="toggle-arrow"; arrow.innerHTML="&#9654;";
    header.appendChild(titleW); header.appendChild(arrow);
    block.appendChild(header);

    const wrapper = document.createElement("div"); wrapper.className="subcat-list"; wrapper.style.paddingLeft="20px";

    (opts.values||[]).forEach(v=>{
      const row = document.createElement("div"); row.className="checkbox-wrapper";
      const chk = document.createElement("input"); chk.type="checkbox"; chk.className="small-checkbox";
      chk.checked = false;
      chk.setAttribute("data-filter", opts.id);
      chk.setAttribute("data-value", String(v));
      const lbl = document.createElement("span"); lbl.textContent = String(v);
      row.appendChild(chk); row.appendChild(lbl);
      wrapper.appendChild(row);
      chk.addEventListener("change", applyFilters);
    });

    block.appendChild(wrapper);
    container.appendChild(block);

    header.addEventListener("click", (e)=>{
      e.stopPropagation();
      const expanded=header.classList.toggle("expanded");
      wrapper.style.display = expanded ? "flex" : "none";
      arrow.innerHTML = expanded ? "&#9660;" : "&#9654;";
    });

    header.classList.add("expanded");
    wrapper.style.display="flex";
    arrow.innerHTML="&#9660;";
  }

  function getActiveValues(filterId){
    const list = Array.from(document.querySelectorAll(`input[data-filter="${filterId}"][type="checkbox"]`));
    return new Set(list.filter(x=>x.checked).map(x=>x.getAttribute("data-value")));
  }

  // Bloques simples
  createSimpleFilterBlock({
    id: "acc_silla",
    emoji: "♿",
    title: "ACCESIBILIDAD SILLA DE RUEDAS",
    values: filtersUnique["acc_silla"] || []
  });
  createSimpleFilterBlock({
    id: "precio",
    emoji: "💶",
    title: "PRECIO",
    values: filtersUnique["precio"] || []
  });

  // ========= NUEVOS BLOQUES DE UMBRAL: VALORACIÓN y NÚMERO DE RESEÑAS =========
  let ratingFilterActive = false;
  let reviewsFilterActive = false;

  function createThresholdFilterBlock(opts){
    const block  = document.createElement("div"); block.className="category-block";
    const header = document.createElement("div"); header.className="category-header";
    const titleW = document.createElement("div"); titleW.className="title";
    const em = document.createElement("span"); em.textContent = opts.emoji || "⭐";
    const nm = document.createElement("span"); nm.textContent = opts.title;
    titleW.appendChild(em); titleW.appendChild(nm);
    const arrow = document.createElement("div"); arrow.className="toggle-arrow"; arrow.innerHTML="&#9654;";
    header.appendChild(titleW); header.appendChild(arrow);
    block.appendChild(header);

    const wrapper = document.createElement("div"); wrapper.className="subcat-list"; wrapper.style.paddingLeft="12px";

    const row = document.createElement("div"); row.className="checkbox-wrapper"; row.style.alignItems="center";
    const label = document.createElement("label"); label.textContent = opts.label + ":"; label.style.minWidth="150px";
    const input = document.createElement("input"); input.type="number"; input.id = opts.inputId;
    if (opts.min !== undefined) input.min = String(opts.min);
    if (opts.max !== undefined) input.max = String(opts.max);
    input.step = String(opts.step ?? 1);
    input.style.flex="1";
    if (opts.placeholder) input.placeholder = opts.placeholder;

    row.appendChild(label); row.appendChild(input);

    const actionsRow = document.createElement("div"); actionsRow.style.display="flex"; actionsRow.style.gap="6px"; actionsRow.style.marginTop="6px";
    const applyBtn = document.createElement("button"); applyBtn.textContent="Aplicar " + opts.shortTitle;
    const clearBtn = document.createElement("button"); clearBtn.textContent="Quitar filtro";
    actionsRow.appendChild(applyBtn); actionsRow.appendChild(clearBtn);

    const hint = document.createElement("div"); hint.style.fontSize="12px"; hint.style.color="#666"; hint.textContent = opts.hint;

    wrapper.appendChild(row); wrapper.appendChild(actionsRow); wrapper.appendChild(hint);
    block.appendChild(wrapper);
    container.appendChild(block);

    header.addEventListener("click", (e)=>{
      e.stopPropagation();
      const expanded=header.classList.toggle("expanded");
      wrapper.style.display = expanded ? "block" : "none";
      arrow.innerHTML = expanded ? "&#9660;" : "&#9654;";
    });

    // abierto por defecto
    header.classList.add("expanded"); wrapper.style.display="block"; arrow.innerHTML="&#9660;";

    applyBtn.addEventListener("click", ()=>{
      const val = input.value.trim();
      if (val === "") { alert("Introduce un valor numérico."); return; }
      const num = parseFloat(val);
      if (Number.isNaN(num)) { alert("Valor no válido."); return; }
      if (opts.min !== undefined && num < opts.min) { alert("El valor debe ser ≥ " + opts.min + "."); return; }
      if (opts.max !== undefined && num > opts.max) { alert("El valor debe ser ≤ " + opts.max + "."); return; }
      if (opts.type === "rating") ratingFilterActive = true;
      if (opts.type === "reviews") reviewsFilterActive = true;
      applyFilters();
    });
    clearBtn.addEventListener("click", ()=>{
      if (opts.type === "rating") ratingFilterActive = false;
      if (opts.type === "reviews") reviewsFilterActive = false;
      applyFilters();
    });
  }

  createThresholdFilterBlock({
    type: "rating",
    emoji: "⭐",
    title: "VALORACIÓN",
    shortTitle: "valoración",
    label: "Mínimo (0–5)",
    inputId: "rating-input",
    min: 0, max: 5, step: 0.1,
    placeholder: "p.ej. 2.3",
    hint: "Muestra lugares con valoración mayor o igual al valor."
  });

  createThresholdFilterBlock({
    type: "reviews",
    emoji: "🧮",
    title: "NÚMERO DE RESEÑAS",
    shortTitle: "n.º reseñas",
    label: "Mínimo (≥ 0)",
    inputId: "reviews-input",
    min: 0, step: 1,
    placeholder: "p.ej. 23",
    hint: "Muestra lugares con un número de reseñas mayor o igual al valor."
  });
  // ========= FIN NUEVOS BLOQUES =========

  createSimpleFilterBlock({
    id: "estado",
    emoji: "🏷️",
    title: "ESTADO DE NEGOCIO",
    values: filtersUnique["estado"] || []
  });
  createSimpleFilterBlock({
    id: "reserva",
    emoji: "📅",
    title: "RESERVA POSIBLE",
    values: filtersUnique["reserva"] || []
  });

  document.getElementById("show-all").addEventListener("click", ()=>{
    document.querySelectorAll("#categories-container input[type=checkbox]").forEach(chk=>{ chk.checked=true; });
    ratingFilterActive = false; reviewsFilterActive = false; // restablece umbrales
    applyFilters();
  });
  document.getElementById("hide-all").addEventListener("click", ()=>{
    document.querySelectorAll("#categories-container input[type=checkbox]").forEach(chk=>{ chk.checked=false; });
    applyFilters();
  });

  function setMarkerVisible(marker, visible){
    const el = marker._icon; if(el) el.style.display = visible ? "" : "none";
    const sh = marker._shadow; if(sh) sh.style.display = visible ? "" : "none";
  }

  function markerMatchesCategoryFilters(marker){
    const catToSubs = marker._catToSubs;
    const mains = Object.keys(catToSubs);
    for (const main of mains){
      const subs = catToSubs[main];
      for (const sub of subs){
        const chk = document.querySelector(`input[data-main='${main}'][data-sub='${sub}']`);
        if (chk && chk.checked) return true;
      }
    }
    return false;
  }

  // === LÓGICA DE FILTRADO (AND entre bloques) ===
  function applyFilters(){
    const favOn = document.getElementById('fav-main-chk').checked;
    if (favOn){ if(!map.hasLayer(favoritesLayer)) favoritesLayer.addTo(map); }
    else { if(map.hasLayer(favoritesLayer)) map.removeLayer(favoritesLayer); }

    // horario activado por botón
    const daySel   = document.getElementById("day-select")?.value ?? "";
    const timeSel  = document.getElementById("time-input")?.value ?? "";
    const timeFilterOn = (typeof hourFilterActive !== "undefined") && hourFilterActive && daySel!=="" && timeSel!=="";
    let dow=null, minute=null;
    if(timeFilterOn){ dow=parseInt(daySel,10); const [HH,MM]=timeSel.split(":").map(x=>parseInt(x,10)); minute=HH*60+MM; }

    // filtros simples
    const accSet     = getActiveValues("acc_silla");
    const precioSet  = getActiveValues("precio");
    const estadoSet  = getActiveValues("estado");
    const reservaSet = getActiveValues("reserva");

    const accActive     = accSet.size > 0;
    const precioActive  = precioSet.size > 0;
    const estadoActive  = estadoSet.size > 0;
    const reservaActive = reservaSet.size > 0;

    // filtros de umbral
    const ratingInput = document.getElementById("rating-input");
    const reviewsInput = document.getElementById("reviews-input");
    const ratingFilterOn = ratingFilterActive && ratingInput && ratingInput.value.trim() !== "";
    const reviewsFilterOn = reviewsFilterActive && reviewsInput && reviewsInput.value.trim() !== "";
    const ratingThreshold = ratingFilterOn ? parseFloat(ratingInput.value) : null;
    const reviewsThreshold = reviewsFilterOn ? parseFloat(reviewsInput.value) : null;

    for(const marker of allMarkers){
      let visible = markerMatchesCategoryFilters(marker);

      if (visible && timeFilterOn){
        let sched = scheduleCache.get(marker._recId);
        if(!sched){ sched = parseSchedule(marker._rec.horario || ""); scheduleCache.set(marker._recId, sched); }
        visible = isOpenAtSchedule(sched, dow, minute);
      }

      if (visible && accActive){
        const v = String(marker._rec.accesibilidad_silla_ruedas);
        visible = accSet.has(v);
      }
      if (visible && precioActive){
        const v = String(marker._rec.precio);
        visible = precioSet.has(v);
      }
      if (visible && estadoActive){
        const v = String(marker._rec.estado_negocio);
        visible = estadoSet.has(v);
      }
      if (visible && reservaActive){
        const v = String(marker._rec.reserva_posible);
        visible = reservaSet.has(v);
      }

      // Umbral de valoración (>=)
      if (visible && ratingFilterOn){
        const v = Number(marker._rec.rating);
        visible = !Number.isNaN(v) && v >= ratingThreshold;
      }
      // Umbral de reseñas (>=)
      if (visible && reviewsFilterOn){
        const v = Number(marker._rec.total_reviews);
        visible = !Number.isNaN(v) && v >= reviewsThreshold;
      }

      setMarkerVisible(marker, visible);
      marker._updateEmoji();
    }
  }

  const panelToggle = document.getElementById("panel-toggle");
  const filterBody  = document.getElementById("filter-body");
  let collapsed = false;
  panelToggle.addEventListener("click", ()=>{ collapsed = !collapsed; filterBody.style.display = collapsed ? "none" : "flex"; panelToggle.innerHTML = collapsed ? "&#9654;" : "&#9660;"; });

  applyFilters();
  // === Pin de la ubicación del usuario (📌) ===
  const userLayer = L.layerGroup().addTo(map);

  function putUserPin(lat, lon, accuracy){
    userLayer.clearLayers();
    const icon = L.divIcon({
      className: "leaflet-div-icon user-pin",
      html: "📌",
      iconSize: [28, 28],
      iconAnchor: [14, 28]
    });
    L.marker([lat, lon], { icon, zIndexOffset: 1000 })
      .addTo(userLayer)
      .bindPopup("Estás aquí");

    if (accuracy && !Number.isNaN(accuracy)) {
      L.circle([lat, lon], { radius: accuracy }).addTo(userLayer);
    }
  }

  // 1) Intenta leer las coords que expone Flask (app.py) en /coords
  fetch("/coords")
    .then(r => r.json())
    .then(c => {
      if (c && typeof c.lat === "number" && typeof c.lon === "number") {
        putUserPin(c.lat, c.lon, null);
        map.setView([c.lat, c.lon], 14);
      }
    })
    .catch(() => { /* sin ruido */ });

  // 2) Como respaldo, mira si el front guardó coords en localStorage
  try {
    const s = localStorage.getItem("client_coords"); // p.ej. {lat, lon, accuracy}
    if (s) {
      const o = JSON.parse(s);
      if (o.lat && o.lon) putUserPin(+o.lat, +o.lon, o.accuracy);
    }
  } catch(e){}

  // 3) Último fallback: geolocalización del navegador
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      pos => {
        const { latitude, longitude, accuracy } = pos.coords;
        putUserPin(latitude, longitude, accuracy);
      },
      () => {},
      { enableHighAccuracy: true, maximumAge: 30000, timeout: 5000 }
    );
  }
});
</script>
"""

m.get_root().html.add_child(folium.Element(js))
m.save(HTML_OUT)
print("✔️  Mapa (filtros actualizados con valoración y número de reseñas) guardado en", HTML_OUT)
