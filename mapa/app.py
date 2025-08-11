import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# 1) Carga de datos
CSV_FILE = "madridactividades.csv"
df = pd.read_csv(CSV_FILE, encoding="utf-8-sig", dtype=str, on_bad_lines="skip")
df["LAT"] = pd.to_numeric(df["LATITUD_TUI"], errors="coerce")
df["LON"] = pd.to_numeric(df["LONGITUD_TUI"], errors="coerce")
df = df.dropna(subset=["LAT", "LON"])
df["CATEGORIA_TUI"] = df["CATEGORIA_TUI"].str.upper().str.strip()

# Extraer subtipos
def get_subtypes(row):
    subs = []
    for i in range(1, 8):
        v = row.get(f"Categoria_{i}")
        if pd.notna(v) and v.strip():
            subs.append(v.strip())
    return subs

df["SUBTYPES"] = df.apply(get_subtypes, axis=1)

# Layout
st.set_page_config(layout="wide")
st.title("Mapa Interactivo - Filtros y Mapa")
col1, col2 = st.columns([1, 3])

# 2) Panel de filtros
with col1:
    st.header("Filtros")
    categories = sorted(df["CATEGORIA_TUI"].unique())
    # Dict para almacenar qué subtipos incluye cada categoría
    selected_filters = {}

    for cat in categories:
        subs = sorted({sub for row in df[df["CATEGORIA_TUI"] == cat]["SUBTYPES"] for sub in row})
        with st.expander(cat):
            all_key = f"{cat}_all"
            select_all = st.checkbox("Seleccionar todos los subtipos", key=all_key)
            chosen_subs = []
            for sub in subs:
                key = f"{cat}_{sub}"
                # Si 'select_all' está marcado, forzamos todos los checkboxes a True y deshabilitados
                if select_all:
                    st.checkbox(sub, key=key, value=True, disabled=True)
                    chosen_subs.append(sub)
                else:
                    if st.checkbox(sub, key=key):
                        chosen_subs.append(sub)
            if chosen_subs:
                selected_filters[cat] = chosen_subs

    # Botones globales
    if st.button("Mostrar todo"):
        selected_filters = {cat: sorted({sub for row in df[df["CATEGORIA_TUI"] == cat]["SUBTYPES"] for sub in row})
                            for cat in categories}
    if st.button("Ocultar todo"):
        selected_filters = {}

# 3) Mapa a la derecha
with col2:
    st.header("Mapa")
    # Filtrado del DataFrame según selección
    if selected_filters:
        def row_matches(r):
            cat = r["CATEGORIA_TUI"]
            return cat in selected_filters and any(sub in selected_filters[cat] for sub in r["SUBTYPES"])

        df_filt = df[df.apply(row_matches, axis=1)]
    else:
        df_filt = df

    # Crear mapa y añadir marcadores
    m = folium.Map(location=[40.4168, -3.7038], zoom_start=11, tiles="CartoDB positron")
    for _, r in df_filt.iterrows():
        folium.Marker([r["LAT"], r["LON"]], popup=r["NOMBRE_TUI"]).add_to(m)

    st_folium(m, width=700, height=500)
