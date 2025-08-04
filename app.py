from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

df = pd.read_csv("Consolidada.csv",encoding="latin1", sep=None, engine="python")

def buscar_info_completa(pregunta, max_resultados=5):
    pregunta = pregunta.lower()
    columnas_relevantes = ['NOMBRE_TUI', 'CATEGORIA_TUI', 'DESCRIPCION_TUI', 'Categoria_1', 'Categoria_2', 'Categoria_3']
    palabras = [palabra for palabra in pregunta.split() if len(palabra) > 2]

    def fila_relevante(row):
        texto = " ".join(str(row[col]).lower() for col in columnas_relevantes if pd.notnull(row[col]))
        return any(palabra in texto for palabra in palabras)

    filas_filtradas = df[df.apply(fila_relevante, axis=1)].head(max_resultados)

    if filas_filtradas.empty:
        return "No encontré información relevante en mis datos."

    resumen = ""
    for _, row in filas_filtradas.iterrows():
        resumen += f"Nombre: {row['NOMBRE_TUI']}\n"
        resumen += f"Categoría: {row['CATEGORIA_TUI']}\n"
        resumen += f"Descripción: {row['DESCRIPCION_TUI']}\n"
        resumen += f"Dirección: {row['DIRECCION_TUI']}\n"
        resumen += f"Horario: {row['HORARIO']}\n"
        resumen += f"Teléfono: {row['TELEFONO']}\n"
        resumen += f"Website: {row['WEBSITE']}\n"
        resumen += "----\n"
    return resumen.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get("messages")
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400

    # Obtiene la última pregunta del usuario
    user_latest = [m['content'] for m in messages if m['role'] == 'user'][-1]

    # Busca info relevante según la última pregunta
    info = buscar_info_completa(user_latest)

    system_prompt = {
        "role": "system",
        "content": "Eres un asistente turístico experto en Madrid, responde basándote en esta información:\n" + info
    }

    full_messages = [system_prompt] + messages

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=full_messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({'response': reply})
    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
