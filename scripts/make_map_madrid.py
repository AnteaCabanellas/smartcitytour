import shelve, time, json
import pandas as pd, folium
from tqdm import tqdm
from geopy.geocoders import Nominatim
from ratelimit import limits, sleep_and_retry

# ------------------------------------------------------------------
CSV_FILE  = "data/BBDDTUI_unida.csv"
HTML_OUT  = "static/mapa/mapa_madrid.html"
CACHE_DB  = "geocode_cache.db"

# diccionario «categoría → emoji»
CAT2EMOJI = {
    "RESTAURANTES":               "🍽️",
    "TIENDAS":                    "👗",
    "PUNTOS DE INTERÉS TURÍSTICO":"🗽",
    "LOCALES DE OCIO NOCTURNO":   "🍸",
    "ESPACIOS DEPORTIVOS":        "⚽",
    "ALOJAMIENTOS":               "🏨",
    "IGLESIAS Y CATEDRALES":      "⛪",
    "INFORMACIÓN TURISMO":        "ℹ️",
    "OFICINAS TURISMO":           "🏢"
}
# ------------------------------------------------------------------

# 1) CARGA Y LIMPIEZA ───────────────────────────────────────
# 1) CARGA Y LIMPIEZA ───────────────────────────────────────
# Lectura robusta del CSV: prueba varias codificaciones y separadores,
# preserva tipos como 'str' y selecciona sólo las columnas objetivo si existen.

TARGET_COLS = [
    "NOMBRE_TUI", "DESCRIPCION_TUI",
    "LATITUD_TUI", "LONGITUD_TUI",
    "DIRECCION_TUI", "CATEGORIA_TUI",
    "TELEFONO", "EMAIL", "HORARIO", "CONTENT_URL",
    "Categoria_1", "Categoria_2", "Categoria_3",
    "Categoria_4", "Categoria_5", "Categoria_6", "Categoria_7"
]

def read_csv_robust(path):
    encodings  = ("utf-8-sig", "utf-8", "cp1252", "latin1")
    separators = (",", ";")
    last_err = None

    for enc in encodings:
        for sep in separators:
            try:
                # 1º intento: leer pidiendo sólo las columnas objetivo
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    dtype=str,
                    on_bad_lines="skip",   # requiere pandas >= 1.3
                    engine="python",
                    usecols=TARGET_COLS
                )
                return df
            except ValueError as ve:
                # Puede fallar porque no encuentra todas las columnas solicitadas.
                # En ese caso, leemos TODO y luego nos quedamos con la intersección.
                try:
                    df_all = pd.read_csv(
                        path,
                        sep=sep,
                        encoding=enc,
                        dtype=str,
                        on_bad_lines="skip",
                        engine="python"
                    )
                    keep = [c for c in TARGET_COLS if c in df_all.columns]
                    if not keep:
                        raise ve  # no hay ninguna columna útil, reintentar con otro (enc, sep)
                    return df_all[keep].copy()
                except Exception as e2:
                    last_err = e2
            except UnicodeDecodeError as ude:
                last_err = ude
            except Exception as e:
                last_err = e

    # Si nada funcionó, relanza el último error para depurar
    raise last_err if last_err else RuntimeError("No se pudo leer el CSV con los encodings/separadores probados.")

# Llamada efectiva
df = read_csv_robust(CSV_FILE)


# Añadimos una columna "ID" única para cada registro
# Aquí uso el índice del DataFrame, pero podría ser otro identificador
# único y estable (por ejemplo CONTENT_URL)
df.reset_index(inplace=True)
df.rename(columns={"index": "ID"}, inplace=True)

df["DESCRIPCION_TUI"] = df["DESCRIPCION_TUI"].fillna("")
df["CATEGORIA_TUI"]   = df["CATEGORIA_TUI"].str.upper().str.strip()

df["LAT"] = pd.to_numeric(df["LATITUD_TUI"], errors="coerce")
df["LON"] = pd.to_numeric(df["LONGITUD_TUI"], errors="coerce")

# 2) FUNCION PARA GEOLOCALIZAR --------------------------------------
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

# 3) COMPLETAR LAT/LON (con cache) ----------------------------------
cache = shelve.open(CACHE_DB)
for i, row in tqdm(df.iterrows(), total=len(df), desc="Geocodificando"):
    if pd.notna(row["LAT"]) and pd.notna(row["LON"]):
        continue
    addr = row["DIRECCION_TUI"]
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

# 4) DESCARTAR FILAS SIN COORDENADAS --------------------------------
df = df.dropna(subset=["LAT", "LON"])
print("Filas válidas →", len(df))

# 5) FUNCION PARA EXTRAER SUBCATEGORIAS -----------------------------

def extract_subcats(row):
    subs = []
    for c in ["Categoria_1", "Categoria_2", "Categoria_3",
              "Categoria_4", "Categoria_5", "Categoria_6", "Categoria_7"]:
        val = row.get(c, "")
        if pd.notna(val) and val.strip():
            subs.append(val.strip())
    return subs

# 6) Preparar datos para JS -----------------------------------------
records = []
for _, r in df.iterrows():
    main_cat = r["CATEGORIA_TUI"] or "SIN CATEGORÍA"
    subcats  = extract_subcats(r) or ["(sin subcategoría)"]
    record = {
        "id": int(r["ID"]),
        "name": r["NOMBRE_TUI"] or "",
        "description": r["DESCRIPCION_TUI"] or "",
        "lat": float(r["LAT"]),
        "lon": float(r["LON"]),
        "address": r["DIRECCION_TUI"] or f"{r['LAT']:.5f}, {r['LON']:.5f}",
        "main_category": main_cat,
        "subcategories": subcats,
        "email": r["EMAIL"] or "",
        "phone": r["TELEFONO"] or "",
        "url": r["CONTENT_URL"] or "",
        "horario": r["HORARIO"] or ""
    }
    records.append(record)

# Extraer subcategorías únicas
cat2subs = {}
for rec in records:
    mc = rec["main_category"]
    cat2subs.setdefault(mc, set())
    cat2subs[mc].update(rec["subcategories"])
# Convertir sets a listas ordenadas
cat2subs = {k: sorted(v) for k, v in cat2subs.items()}

# 7) CREA EL MAPA BASE ---------------------------------------------
m = folium.Map(location=[40.4168, -3.7038], zoom_start=11, tiles="CartoDB positron")

# 8) Inyectar datos y control de filtros + FAVORITOS ---------------

def safe_json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False).replace("</script>", "<\\/script>")

data_json      = safe_json_dumps(records)
cat2subs_json  = safe_json_dumps(cat2subs)
emoji_json     = safe_json_dumps(CAT2EMOJI)

js = """
<style>
/* --- PANEL DE FILTROS -------------------------------------------------- */
#filter-panel { position:absolute; top:10px; right:10px; width:300px; max-height:85vh; font-family:sans-serif; z-index:1000; }
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
.subcat-list { margin-left:4px; margin-top:4px; display:none; flex-direction:column; gap:2px; max-height:300px; overflow:auto; padding-left:10px; }
.toggle-arrow { width:16px; display:inline-block; transform:rotate(0deg); transition:transform .2s ease; font-weight:bold; }
.category-header.expanded .toggle-arrow { transform:rotate(90deg); }
.checkbox-wrapper { display:flex; align-items:center; gap:6px; padding:2px 0; }
.small-checkbox { width:16px; height:16px; }
/* --- ICONO CORAZÓN ----------------------------------------------------- */
.fav-heart { cursor:pointer; font-size:18px; user-select:none; }
/* --- ICONO DESCARGA ---------------------------------------------------- */
.download-icon { cursor:pointer; font-size:18px; user-select:none; margin-left:auto; padding:2px 6px; border-radius:4px; transition:background .15s ease; }
.download-icon:hover { background:#e0e0ea; }
/* --- ESTILOS PARA RESEÑAS ---------------------------------------------- */
button[id^="review-btn-"]:hover { background:#45a049 !important; }
textarea[id^="review-input-"]:focus { outline:none; border-color:#4CAF50; }
.review-like { transition: transform 0.1s ease; }
.review-like:hover { transform: scale(1.2); }
.review-edit { transition: transform 0.1s ease; }
.review-edit:hover { transform: scale(1.2); }
.review-delete { transition: transform 0.1s ease; }
.review-delete:hover { transform: scale(1.2); }
.save-edit:hover { background:#45a049 !important; }
.cancel-edit:hover { background:#777 !important; }
</style>

<div id=\"filter-panel\">\n  
<div class=\"panel leaflet-bar\">\n   
<div id=\"filter-header\">\n      
<div class=\"title-wrapper\"><h3>Filtros</h3></div>\n      
<div id=\"panel-toggle\">&#9660;</div>\n    
</div>\n    
<div id=\"filter-body\">\n      
<div class=\"buttons\">\n        
<button id=\"show-all\">Mostrar todo</button>\n        
<button id=\"hide-all\">Ocultar todo</button>\n      
</div>\n      
<div id=\"categories-container\" style=\"overflow:auto; flex:1; margin-top:4px; padding-right:4px;\"></div>\n    
</div>\n  
</div>\n</div>

<script>
/****************** AUXILIARES MAPA *************************/
function getFoliumMap() {
  if (window._folium_map) return window._folium_map;
  for (const k in window) {
    try { if (window[k] instanceof L.Map) { window._folium_map = window[k]; return window[k]; } } catch {}
  }
  return null;
}
function withMap(cb, tries=0) {
  const m = getFoliumMap();
  if (m) cb(m); else if (tries < 15) setTimeout(()=>withMap(cb, tries+1),100);
}
/****************** DATOS DESDE PYTHON **********************/
const rawData   = """ + data_json + """;
const cat2subs  = """ + cat2subs_json + """;
const cat2emoji = """ + emoji_json + """;
const FAVORITES_KEY = "favorites_mapa";
const REVIEWS_KEY = "reviews_mapa";

/****************** ESTADO FAVORITOS ***********************/
let favorites = new Set(JSON.parse(localStorage.getItem(FAVORITES_KEY) || "[]"));
function saveFavorites() { localStorage.setItem(FAVORITES_KEY, JSON.stringify(Array.from(favorites))); }

/****************** ESTADO RESEÑAS ************************/
let reviews = JSON.parse(localStorage.getItem(REVIEWS_KEY) || "{}");
function saveReviews() { localStorage.setItem(REVIEWS_KEY, JSON.stringify(reviews)); }

function addReview(recId, review) {
    if (!reviews[recId]) reviews[recId] = [];
    reviews[recId].push({
        id: Date.now(), // ID único para cada reseña
        text: review,
        date: new Date().toLocaleString('es-ES'),
        likes: 0,
        liked: false
    });
    saveReviews();
}

function deleteReview(recId, reviewId) {
    if (reviews[recId]) {
        reviews[recId] = reviews[recId].filter(r => r.id !== reviewId);
        if (reviews[recId].length === 0) delete reviews[recId];
        saveReviews();
    }
}

function editReview(recId, reviewId, newText) {
    if (reviews[recId]) {
        const review = reviews[recId].find(r => r.id === reviewId);
        if (review) {
            review.text = newText;
            review.date = new Date().toLocaleString('es-ES') + ' (editado)';
            saveReviews();
        }
    }
}

function toggleLike(recId, reviewId) {
    if (reviews[recId]) {
        const review = reviews[recId].find(r => r.id === reviewId);
        if (review) {
            if (review.liked) {
                review.likes--;
                review.liked = false;
            } else {
                review.likes++;
                review.liked = true;
            }
            saveReviews();
        }
    }
}

/****************** CAPAS LEAFLET **************************/
const layerByCatSub = {};    // capas normales por categoría/subcategoría
let   favoritesLayer;        // capa exclusiva de favoritos

withMap((map)=>{

  /*********** CREAR CAPA FAVORITOS ************************/ 
  favoritesLayer = L.layerGroup().addTo(map);

  /*********** CREAR MARCADORES ****************************/
  for (const rec of rawData) {
      const main = rec.main_category || "(sin categoría)";
      const subs = rec.subcategories.length ? rec.subcategories : ["(sin subcategoría)"];
      const emoji = cat2emoji[main] || "📍";
      const isFav = favorites.has(rec.id);
      const favSymbol = isFav ? "♥" : "♡";

      // --- popup ---
      const urlHTML = rec.url ? `<a href="${rec.url}" target="_blank" rel="noopener noreferrer">${rec.url}</a>` : '—';
      
      // Obtener reseñas existentes
      const placeReviews = reviews[rec.id] || [];
      let reviewsHTML = '';
      if (placeReviews.length > 0) {
          reviewsHTML = placeReviews.map(r => `
              <div class="review-item" data-review-id="${r.id}" style="background:#f5f5f5; padding:8px; margin:4px 0; border-radius:4px; border-left:3px solid #4CAF50; position:relative;">
                  <div style="position:absolute; top:8px; right:8px; display:flex; gap:8px;">
                      <span class="review-like" data-rec-id="${rec.id}" data-review-id="${r.id}" 
                            style="cursor:pointer; font-size:16px; user-select:none;" 
                            title="Me gusta">
                          ${r.liked ? '👍' : '👍'}<span style="font-size:12px; color:#666; margin-left:2px;">${r.likes || 0}</span>
                      </span>
                      <span class="review-edit" data-rec-id="${rec.id}" data-review-id="${r.id}" 
                            style="cursor:pointer; font-size:16px; user-select:none;" 
                            title="Editar">✏️</span>
                      <span class="review-delete" data-rec-id="${rec.id}" data-review-id="${r.id}" 
                            style="cursor:pointer; font-size:16px; user-select:none;" 
                            title="Eliminar">🗑️</span>
                  </div>
                  <div style="font-size:12px; color:#666; margin-bottom:4px; padding-right:100px;">${r.date}</div>
                  <div class="review-text" style="font-size:13px; line-height:1.4; padding-right:100px;">${r.text}</div>
                  <div class="review-edit-area" style="display:none; margin-top:8px;">
                      <textarea class="edit-textarea" style="width:100%; min-height:50px; padding:6px; border:1px solid #4CAF50; border-radius:4px; font-size:13px; box-sizing:border-box;">${r.text}</textarea>
                      <div style="margin-top:6px; display:flex; gap:6px;">
                          <button class="save-edit" style="padding:4px 12px; background:#4CAF50; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px;">Guardar</button>
                          <button class="cancel-edit" style="padding:4px 12px; background:#999; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px;">Cancelar</button>
                      </div>
                  </div>
              </div>
          `).join('');
      } else {
          reviewsHTML = '<div style="color:#999; font-style:italic; font-size:13px">No hay reseñas todavía</div>';
      }
      
      const popupContent = `
        <div style="font-size:14px; max-width:850px; max-height:700px; overflow:auto">
          <b>${rec.name}</b>&nbsp;&nbsp;<span class="fav-heart" data-id="${rec.id}">${favSymbol}</span><br>
          <small>${rec.address}</small><br><br>
          ${rec.description}<br><br>
          <strong>Email:</strong> ${rec.email || '—'}<br>
          <strong>URL:</strong> ${urlHTML}<br>
          <strong>Teléfono:</strong> ${rec.phone || '—'}<br>
          <strong>Horario:</strong> ${rec.horario || '—'}<br><br>
          <strong>Reseñas:</strong>
          <div style="margin-top:8px; max-height:250px; overflow-y:auto" id="reviews-container-${rec.id}">
              ${reviewsHTML}
          </div>
          <div style="margin-top:10px; border-top:1px solid #ddd; padding-top:10px">
              <textarea id="review-input-${rec.id}" 
                       placeholder="Escribe tu reseña aquí..." 
                       style="width:100%; min-height:60px; padding:8px; border:1px solid #ccc; 
                              border-radius:4px; resize:vertical; font-family:sans-serif; 
                              font-size:13px; box-sizing:border-box"></textarea>
              <button id="review-btn-${rec.id}" 
                      style="margin-top:6px; padding:6px 16px; background:#4CAF50; color:white; 
                             border:none; border-radius:4px; cursor:pointer; font-size:13px">
                  Añadir reseña
              </button>
          </div>
        </div>`;

      // --- icono ---
      const icon = L.divIcon({ html:`<div style="font-size:22px; line-height:1">${emoji}</div>`, className:"" });
      const marker = L.marker([rec.lat, rec.lon], { icon }).bindPopup(popupContent, {maxWidth:900,maxHeight:800});

      // --- almacenar referencia a marker para favoritos ----
      marker._recId = rec.id;

      // --- añadir a estructuras por categoría --------------
      for (const sub of subs) {
          layerByCatSub[main] ??= {};
          layerByCatSub[main][sub] ??= L.layerGroup();
          layerByCatSub[main][sub].addLayer(marker);
      }

      // --- si es favorito actual, añadir a favoritesLayer ---
      if (isFav) favoritesLayer.addLayer(marker);
  }

  // Añadir todas las capas normales por defecto
  for (const main in layerByCatSub) {
      for (const sub in layerByCatSub[main]) {
          layerByCatSub[main][sub].addTo(map);
      }
  }

  /****************** POPUP HEART INTERACTIVO ********************/
  map.on('popupopen', (e)=>{
      const pop = e.popup?.getElement();
      if (!pop) return;
      
      // --- Manejo del corazón de favoritos ---
      const heart = pop.querySelector('.fav-heart');
      if (heart) {
          const recId = parseInt(heart.getAttribute('data-id'));
          heart.addEventListener('click', ()=>{
              const isFav = favorites.has(recId);
              if (isFav) {
                  favorites.delete(recId);
                  heart.textContent = "♡";
                  // quitar del layer de favoritos
                  favoritesLayer.eachLayer(l=>{ if (l._recId===recId) favoritesLayer.removeLayer(l); });
              } else {
                  favorites.add(recId);
                  heart.textContent = "♥";
                  // buscar el marker y añadirlo al layer de favoritos
                  for (const main in layerByCatSub) {
                      for (const sub in layerByCatSub[main]) {
                          layerByCatSub[main][sub].eachLayer(l=>{ if (l._recId===recId) favoritesLayer.addLayer(l); });
                      }
                  }
              }
              saveFavorites();
              updateFavoritesUI();
              applyFilters();
          }, { once:false });
      }
      
      // --- Manejo de reseñas ---
      const reviewBtn = pop.querySelector(`[id^="review-btn-"]`);
      if (reviewBtn) {
          const recId = parseInt(reviewBtn.id.split('-')[2]);
          const reviewInput = pop.querySelector(`#review-input-${recId}`);
          const reviewsContainer = pop.querySelector(`#reviews-container-${recId}`);
          
          reviewBtn.addEventListener('click', ()=>{
              const reviewText = reviewInput.value.trim();
              if (reviewText) {
                  // Añadir la reseña
                  const reviewId = Date.now();
                  if (!reviews[recId]) reviews[recId] = [];
                  const newReview = {
                      id: reviewId,
                      text: reviewText,
                      date: new Date().toLocaleString('es-ES'),
                      likes: 0,
                      liked: false
                  };
                  reviews[recId].push(newReview);
                  saveReviews();
                  
                  // Si es la primera reseña, quitar el mensaje de "no hay reseñas"
                  if (reviews[recId].length === 1) {
                      reviewsContainer.innerHTML = '';
                  }
                  
                  // Añadir la nueva reseña al contenedor
                  const reviewDiv = document.createElement('div');
                  reviewDiv.className = 'review-item';
                  reviewDiv.setAttribute('data-review-id', reviewId);
                  reviewDiv.style.cssText = 'background:#f5f5f5; padding:8px; margin:4px 0; border-radius:4px; border-left:3px solid #4CAF50; position:relative;';
                  reviewDiv.innerHTML = `
                      <div style="position:absolute; top:8px; right:8px; display:flex; gap:8px;">
                          <span class="review-like" data-rec-id="${recId}" data-review-id="${reviewId}" 
                                style="cursor:pointer; font-size:16px; user-select:none;" 
                                title="Me gusta">
                              👍<span style="font-size:12px; color:#666; margin-left:2px;">0</span>
                          </span>
                          <span class="review-edit" data-rec-id="${recId}" data-review-id="${reviewId}" 
                                style="cursor:pointer; font-size:16px; user-select:none;" 
                                title="Editar">✏️</span>
                          <span class="review-delete" data-rec-id="${recId}" data-review-id="${reviewId}" 
                                style="cursor:pointer; font-size:16px; user-select:none;" 
                                title="Eliminar">🗑️</span>
                      </div>
                      <div style="font-size:12px; color:#666; margin-bottom:4px; padding-right:100px;">${newReview.date}</div>
                      <div class="review-text" style="font-size:13px; line-height:1.4; padding-right:100px;">${newReview.text}</div>
                      <div class="review-edit-area" style="display:none; margin-top:8px;">
                          <textarea class="edit-textarea" style="width:100%; min-height:50px; padding:6px; border:1px solid #4CAF50; border-radius:4px; font-size:13px; box-sizing:border-box;">${newReview.text}</textarea>
                          <div style="margin-top:6px; display:flex; gap:6px;">
                              <button class="save-edit" style="padding:4px 12px; background:#4CAF50; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px;">Guardar</button>
                              <button class="cancel-edit" style="padding:4px 12px; background:#999; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px;">Cancelar</button>
                          </div>
                      </div>
                  `;
                  reviewsContainer.appendChild(reviewDiv);
                  
                  // Limpiar el campo de texto
                  reviewInput.value = '';
                  
                  // Hacer scroll al final de las reseñas
                  reviewsContainer.scrollTop = reviewsContainer.scrollHeight;
                  
                  // Añadir event listeners a los nuevos botones
                  attachReviewEventListeners(reviewDiv, recId, reviewId);
              } else {
                  alert('Por favor, escribe una reseña antes de enviar.');
              }
          }, { once:false });
      }
      
      // --- Manejo de botones de reseñas existentes ---
      pop.querySelectorAll('.review-item').forEach(reviewItem => {
          const reviewId = parseInt(reviewItem.getAttribute('data-review-id'));
          const recId = parseInt(reviewItem.querySelector('.review-delete').getAttribute('data-rec-id'));
          attachReviewEventListeners(reviewItem, recId, reviewId);
      });
  });
  
  // Función auxiliar para adjuntar event listeners a los botones de reseña
  function attachReviewEventListeners(reviewDiv, recId, reviewId) {
      // Like
      const likeBtn = reviewDiv.querySelector('.review-like');
      if (likeBtn) {
          likeBtn.addEventListener('click', ()=>{
              toggleLike(recId, reviewId);
              const review = reviews[recId]?.find(r => r.id === reviewId);
              if (review) {
                  likeBtn.innerHTML = `👍<span style="font-size:12px; color:#666; margin-left:2px;">${review.likes}</span>`;
                  likeBtn.style.opacity = review.liked ? '1' : '0.6';
              }
          });
      }
      
      // Delete
      const deleteBtn = reviewDiv.querySelector('.review-delete');
      if (deleteBtn) {
          deleteBtn.addEventListener('click', ()=>{
              if (confirm('¿Estás seguro de que quieres eliminar esta reseña?')) {
                  deleteReview(recId, reviewId);
                  reviewDiv.remove();
                  // Si no quedan reseñas, mostrar mensaje
                  const container = document.querySelector(`#reviews-container-${recId}`);
                  if (container && container.children.length === 0) {
                      container.innerHTML = '<div style="color:#999; font-style:italic; font-size:13px">No hay reseñas todavía</div>';
                  }
              }
          });
      }
      
      // Edit
      const editBtn = reviewDiv.querySelector('.review-edit');
      const textDiv = reviewDiv.querySelector('.review-text');
      const editArea = reviewDiv.querySelector('.review-edit-area');
      const editTextarea = reviewDiv.querySelector('.edit-textarea');
      const saveBtn = reviewDiv.querySelector('.save-edit');
      const cancelBtn = reviewDiv.querySelector('.cancel-edit');
      
      if (editBtn && textDiv && editArea && editTextarea) {
          editBtn.addEventListener('click', ()=>{
              textDiv.style.display = 'none';
              editArea.style.display = 'block';
              editTextarea.focus();
          });
          
          if (cancelBtn) {
              cancelBtn.addEventListener('click', ()=>{
                  textDiv.style.display = 'block';
                  editArea.style.display = 'none';
                  const review = reviews[recId]?.find(r => r.id === reviewId);
                  if (review) editTextarea.value = review.text;
              });
          }
          
          if (saveBtn) {
              saveBtn.addEventListener('click', ()=>{
                  const newText = editTextarea.value.trim();
                  if (newText) {
                      editReview(recId, reviewId, newText);
                      textDiv.textContent = newText;
                      textDiv.style.display = 'block';
                      editArea.style.display = 'none';
                      // Actualizar fecha con indicador de edición
                      const dateDiv = reviewDiv.querySelector('div[style*="color:#666"]');
                      if (dateDiv) {
                          dateDiv.textContent = new Date().toLocaleString('es-ES') + ' (editado)';
                      }
                  } else {
                      alert('La reseña no puede estar vacía.');
                  }
              });
          }
      }
  }

  /*********** EXPORTAR FAVORITOS A CSV ***********/
  function exportFavorites() {
    if (favorites.size === 0) {
      alert("No has marcado favoritos todavía.");
      return;
    }
    // Cabeceras que quieres en el CSV — ajusta a tus necesidades
    const headers = [
      "id","name","address","lat","lon","main_category",
      "subcategories","email","phone","url","horario","description","reseñas"
    ];

    // Construir filas con los registros que estén en favorites
    const rows = rawData
      .filter(r => favorites.has(r.id))
      .map(r => {
        // Obtener las reseñas para este lugar
        const placeReviews = reviews[r.id] || [];
        const reviewsText = placeReviews.map(rev => 
          `${rev.date}: ${rev.text} (${rev.likes} likes)`
        ).join(" | ");
        
        return headers.map(h => {
          let val;
          if (h === "subcategories") {
            val = (r[h] || []).join("|");
          } else if (h === "reseñas") {
            val = reviewsText;
          } else {
            val = r[h];
          }
          // escapado CSV muy simple: comillas dobles y separar con ;
          return `"${String(val || "").replace(/\"/g,'\"\"')}"`;
        }).join(";");
      });

    // Ensamblar CSV
    const csv = [headers.join(";"), ...rows].join("\\r\\n");

    // Crear blob + disparar descarga
    const blob = new Blob([csv], {type: "text/csv;charset=utf-8"});
    const url  = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "favoritos.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /****************** PANEL DE FILTROS ***************************/
  const container = document.getElementById("categories-container");

  /* --- FAVORITOS BLOQUE ------------------------------------- */
  const favBlock = document.createElement("div");
  favBlock.className = "category-block";
  const favHeader = document.createElement("div");
  favHeader.className = "category-header";
  const favTitleW = document.createElement("div");
  favTitleW.className = "title";
  const favChk = document.createElement("input");
  favChk.type = "checkbox";
  favChk.className = "small-checkbox";
  favChk.checked = true;
  favChk.id = "fav-main-chk";
  const favEmoji = document.createElement("span");
  favEmoji.textContent = "❤";
  const favName = document.createElement("span");
  favName.textContent = "FAVORITOS";
  
  // Añadir icono de descarga
  const downloadIcon = document.createElement("span");
  downloadIcon.className = "download-icon";
  downloadIcon.textContent = "📥";
  downloadIcon.title = "Descargar favoritos como CSV";
  
  favTitleW.appendChild(favChk); 
  favTitleW.appendChild(favEmoji); 
  favTitleW.appendChild(favName);
  favTitleW.appendChild(downloadIcon); // Añadir el icono aquí
  favHeader.appendChild(favTitleW);
  favBlock.appendChild(favHeader);
  container.appendChild(favBlock);

  favChk.addEventListener('change', applyFilters);
  
  // Evento click para el icono de descarga
  downloadIcon.addEventListener('click', (e) => {
    e.stopPropagation(); // Evitar que se propague al header
    exportFavorites();
  });

  function updateFavoritesUI() {
      // ahora mismo solo necesitamos actualizar el checkbox indeterminate si no hay favoritos
      if (favorites.size===0) {
          favChk.indeterminate = false;
          favChk.checked = false;
      }
  }
  updateFavoritesUI();

  /* --- BLOQUE DE CATEGORÍAS (contenedor principal) ------------ */
  const catMainBlock = document.createElement("div");
  catMainBlock.className = "category-block";
  
  const catMainHeader = document.createElement("div");
  catMainHeader.className = "category-header";
  
  const catMainTitleW = document.createElement("div");
  catMainTitleW.className = "title";
  
  const catMainChk = document.createElement("input");
  catMainChk.type = "checkbox";
  catMainChk.className = "small-checkbox";
  catMainChk.checked = true;
  catMainChk.id = "cat-main-chk";
  
  const catMainEmoji = document.createElement("span");
  catMainEmoji.textContent = "📁";
  
  const catMainName = document.createElement("span");
  catMainName.textContent = "CATEGORÍAS";
  
  catMainTitleW.appendChild(catMainChk);
  catMainTitleW.appendChild(catMainEmoji);
  catMainTitleW.appendChild(catMainName);
  
  const catMainArrow = document.createElement("div");
  catMainArrow.className = "toggle-arrow";
  catMainArrow.innerHTML = "&#9654;"; // ►
  
  catMainHeader.appendChild(catMainTitleW);
  catMainHeader.appendChild(catMainArrow);
  catMainBlock.appendChild(catMainHeader);
  
  // Contenedor para todas las categorías
  const categoriesWrapper = document.createElement("div");
  categoriesWrapper.className = "subcat-list";
  categoriesWrapper.style.paddingLeft = "20px";
  
  /* --- BLOQUES NORMALES (ahora dentro de categoriesWrapper) --- */
  for (const main of Object.keys(cat2subs)) {
      const block = document.createElement("div");
      block.style.marginTop = "8px";

      const header = document.createElement("div");
      header.className = "category-header";
      header.style.padding = "4px 6px";
      const titleWrapper = document.createElement("div");
      titleWrapper.className = "title";

      const chkCat = document.createElement("input");
      chkCat.type = "checkbox";
      chkCat.className = "small-checkbox";
      chkCat.setAttribute("data-main", main);
      chkCat.checked = true;

      const emojiSpan = document.createElement("span");
      emojiSpan.textContent = cat2emoji[main] || "📍";

      const nameSpan = document.createElement("span");
      nameSpan.textContent = main;

      titleWrapper.appendChild(chkCat);
      titleWrapper.appendChild(emojiSpan);
      titleWrapper.appendChild(nameSpan);

      const arrow = document.createElement("div");
      arrow.className = "toggle-arrow";
      arrow.innerHTML = "&#9654;"; // ►

      header.appendChild(titleWrapper);
      header.appendChild(arrow);
      block.appendChild(header);

      const sublist = document.createElement("div");
      sublist.className = "subcat-list";
      sublist.style.paddingLeft = "20px";

      for (const sub of cat2subs[main]) {
          const row = document.createElement("div");
          row.className = "checkbox-wrapper";

          const chkSub = document.createElement("input");
          chkSub.type = "checkbox";
          chkSub.checked = true;
          chkSub.setAttribute("data-main", main);
          chkSub.setAttribute("data-sub", sub);

          const lbl = document.createElement("span");
          lbl.textContent = sub;

          row.appendChild(chkSub);
          row.appendChild(lbl);
          sublist.appendChild(row);

          chkSub.addEventListener("change", ()=>{
              updateCategoryCheckboxState(main);
              updateMainCategoriesCheckbox();
              applyFilters();
          });
      }

      if (cat2subs[main].length === 0) {
          const row = document.createElement("div");
          row.textContent = "(sin subcategoría)";
          sublist.appendChild(row);
      }

      block.appendChild(sublist);
      categoriesWrapper.appendChild(block);

      header.addEventListener("click", (e)=>{
          e.stopPropagation();
          const expanded = header.classList.toggle("expanded");
          if (expanded) { sublist.style.display="flex"; arrow.innerHTML = "&#9660;"; }
          else           { sublist.style.display="none"; arrow.innerHTML = "&#9654;"; }
      });

      chkCat.addEventListener("change", ()=>{
          const check = chkCat.checked;
          sublist.querySelectorAll("input[type=checkbox]").forEach(si=>{ si.checked = check; });
          chkCat.indeterminate = false;
          updateMainCategoriesCheckbox();
          applyFilters();
      });
  }
  
  catMainBlock.appendChild(categoriesWrapper);
  container.appendChild(catMainBlock);
  
  // Toggle para el bloque principal de CATEGORÍAS
  catMainHeader.addEventListener("click", ()=>{
      const expanded = catMainHeader.classList.toggle("expanded");
      if (expanded) { 
          categoriesWrapper.style.display="block"; 
          catMainArrow.innerHTML = "&#9660;"; 
      } else { 
          categoriesWrapper.style.display="none"; 
          catMainArrow.innerHTML = "&#9654;"; 
      }
  });
  
  // Checkbox principal de CATEGORÍAS
  catMainChk.addEventListener("change", ()=>{
      const check = catMainChk.checked;
      categoriesWrapper.querySelectorAll("input[type=checkbox]").forEach(chk=>{ 
          chk.checked = check; 
          chk.indeterminate = false;
      });
      catMainChk.indeterminate = false;
      applyFilters();
  });

  // Botones globales
  document.getElementById("show-all").addEventListener("click", ()=>{
      document.querySelectorAll("#categories-container input[type=checkbox]").forEach(chk=>{ chk.checked=true; chk.indeterminate=false; });
      updateMainCategoriesCheckbox();
      applyFilters();
  });
  document.getElementById("hide-all").addEventListener("click", ()=>{
      document.querySelectorAll("#categories-container input[type=checkbox]").forEach(chk=>{ chk.checked=false; chk.indeterminate=false; });
      updateMainCategoriesCheckbox();
      applyFilters();
  });

  // Helpers de UI (definidos después para que conozcan funciones) -----
  function updateCategoryCheckboxState(main) {
      const subs = Object.keys(layerByCatSub[main] || {});
      const catCheckbox = document.querySelector(`input[data-main='${main}']`);
      const subChecks = subs.map(s=> document.querySelector(`input[data-main='${main}'][data-sub='${s}']`));
      const checkedCount = subChecks.filter(c=>c && c.checked).length;
      if (!catCheckbox) return;
      if (checkedCount===0)       { catCheckbox.checked=false; catCheckbox.indeterminate=false; }
      else if (checkedCount===subChecks.length) { catCheckbox.checked=true;  catCheckbox.indeterminate=false; }
      else                        { catCheckbox.checked=false; catCheckbox.indeterminate=true;  }
  }
  
  // Función para actualizar el estado del checkbox principal de CATEGORÍAS
  function updateMainCategoriesCheckbox() {
      const categoriesWrapper = document.querySelector(".subcat-list");
      const catMainChk = document.getElementById("cat-main-chk");
      if (!categoriesWrapper || !catMainChk) return;
      
      const allChecks = categoriesWrapper.querySelectorAll("input[type=checkbox]");
      const checkedCount = Array.from(allChecks).filter(c=>c.checked).length;
      
      if (checkedCount === 0) { 
          catMainChk.checked = false; 
          catMainChk.indeterminate = false; 
      } else if (checkedCount === allChecks.length) { 
          catMainChk.checked = true;  
          catMainChk.indeterminate = false; 
      } else { 
          catMainChk.checked = false; 
          catMainChk.indeterminate = true;  
      }
  }

  /****************** APLICAR FILTROS *******************************/
  function applyFilters() {
      /* --- FAVORITOS --- */
      if (favChk.checked) { if (!map.hasLayer(favoritesLayer)) map.addLayer(favoritesLayer); }
      else                { if (map.hasLayer(favoritesLayer))  map.removeLayer(favoritesLayer); }

      /* --- CAPAS NORMALES --- */
      for (const main in layerByCatSub) {
          for (const sub in layerByCatSub[main]) {
              const chk = document.querySelector(`input[data-main='${main}'][data-sub='${sub}']`);
              const layer = layerByCatSub[main][sub];
              if (chk && chk.checked) { if (!map.hasLayer(layer)) map.addLayer(layer); }
              else                    { if (map.hasLayer(layer))  map.removeLayer(layer); }
          }
      }
  }

  // Toggle global del panel
  const panelToggle = document.getElementById("panel-toggle");
  const filterBody  = document.getElementById("filter-body");
  let collapsed = false;
  panelToggle.addEventListener("click", ()=>{
      collapsed = !collapsed;
      if (collapsed) { filterBody.style.display="none"; panelToggle.innerHTML="&#9654;"; }
      else           { filterBody.style.display="flex"; panelToggle.innerHTML="&#9660;"; }
  });

  // Primera aplicación de filtros
  applyFilters();
});
</script>
"""

filter_control = folium.Element(js)
m.get_root().html.add_child(filter_control)

# 9) Guardar HTML ---------------------------------------------------
m.save(HTML_OUT)
print("✔️  Mapa con panel de filtros + favoritos guardado en", HTML_OUT)