from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
import json
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Carga datos CSV ---
try:
    df = pd.read_csv("data/Consolidada.csv", encoding="latin1", sep=None, engine="python")
    df.columns = df.columns.str.strip().str.upper()
except Exception as e:
    print("❌ Error al cargar CSV:", e)
    df = pd.DataFrame()

columnas_relevantes = [col for col in ['NOMBRE_TUI', 'CATEGORIA_TUI', 'DESCRIPCION_TUI', 'CATEGORIA_1', 'CATEGORIA_2', 'CATEGORIA_3'] if col in df.columns]

# --- Buscador local flexible ---
def buscar_info_completa(pregunta, max_resultados=5):
    pregunta = pregunta.lower()
    palabras = [p for p in pregunta.split() if len(p) > 2]

    def fila_relevante(row):
        texto = " ".join(str(row.get(col, "")).lower() for col in columnas_relevantes if pd.notnull(row.get(col)))
        direccion = str(row.get("DIRECCION_TUI", "")).lower()
        categoria = str(row.get("CATEGORIA_TUI", "")).lower() if pd.notnull(row.get("CATEGORIA_TUI")) else ""
        # Solo filtramos por ubicación para mantener flexibilidad en categoría
        if "madrid" not in direccion and "españa" not in direccion:
            return False
        # Buscamos si alguna palabra está en el texto o en la categoría
        return any(p in texto or p in categoria for p in palabras)

    filas_filtradas = df[df.apply(fila_relevante, axis=1)].head(max_resultados)

    if filas_filtradas.empty:
        return None

    resumen = ""
    for _, row in filas_filtradas.iterrows():
        resumen += f"\n🏛️ *{row.get('NOMBRE_TUI', 'Sin nombre')}* ({row.get('CATEGORIA_TUI', 'Sin categoría')})\n"
        resumen += f"📝 {row.get('DESCRIPCION_TUI', 'Sin descripción')}\n"
        if pd.notnull(row.get('DIRECCION_TUI')): resumen += f"📍 Dirección: {row['DIRECCION_TUI']}\n"
        if pd.notnull(row.get('HORARIO')): resumen += f"🕒 Horario: {row['HORARIO']}\n"
        if pd.notnull(row.get('TELEFONO')): resumen += f"📞 Teléfono: {row['TELEFONO']}\n"
        if pd.notnull(row.get('WEBSITE')): resumen += f"🔗 Web: {row['WEBSITE']}\n"
        resumen += "--------------------------\n"
    return resumen.strip()

# --- Rutas ---

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

    info_local = buscar_info_completa(user_latest)
    
    if info_local is None:
        system_prompt_content = (
            "Eres un asistente experto en turismo y actividades en Madrid. "
            "Responde de forma amable y útil a las preguntas del usuario."
        )
    else:
        system_prompt_content = (
            "Eres un asistente experto en turismo y actividades en Madrid. "
            "Aquí tienes información local relevante que debes usar para responder SOLO basándote en ella:\n"
            + info_local
        )

    system_prompt = {
        "role": "system",
        "content": system_prompt_content
    }

    full_messages = [system_prompt] + messages

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=full_messages,
            temperature=0.7,
            max_tokens=500,
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({'response': reply})
    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({'error': str(e)}), 500

# --- Función para preparar JSONL para fine-tuning ---
def preparar_jsonl_desde_df(df, output_path="fine_tune_data.jsonl"):
    required_cols = ['NOMBRE_TUI', 'DESCRIPCION_TUI']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Columnas requeridas {required_cols} no están en el CSV.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df[required_cols].dropna().iterrows():
            prompt = f"Describe el sitio turístico llamado: {row['NOMBRE_TUI'].strip()}\n"
            completion = " " + row['DESCRIPCION_TUI'].strip() + "\n"
            json_line = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
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
            model="gpt-3.5-turbo"
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
            "updated_at": status_resp.updated_at
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
