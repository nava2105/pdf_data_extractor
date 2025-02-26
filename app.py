import os
import re
import csv
import io
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pdf_processor import allowed_file, extract_text_chunks
from ai_processor import configure_gemini_api, generate_text_embeddings, generate_ai_response

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
JSON_STORE = 'embeddings_store.json'
RESPONSES_STORE = 'responses_store.json'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def load_json_store(filepath):
    """Loads data from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    return {}


def save_json_store(filepath, data):
    """Saves data to a JSON file."""
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


# Load stored embeddings and responses
embeddings_store = load_json_store(JSON_STORE)
responses_store = load_json_store(RESPONSES_STORE)

# Configure AI API
configure_gemini_api()


@app.route('/')
def index():
    return render_template("index.html", files=list(embeddings_store.keys()), responses=responses_store)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text_chunks = extract_text_chunks(file_path)
        embeddings = generate_text_embeddings(text_chunks)
        if embeddings:
            embeddings_store[filename] = {'chunks': embeddings}
            save_json_store(JSON_STORE, embeddings_store)
        return redirect(url_for('index'))
    return redirect(request.url)


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    filename = data.get('filename', '')

    if filename not in embeddings_store:
        return jsonify({'error': 'Document not found'}), 400

    document_data = embeddings_store[filename]
    context = "\n".join([chunk[0] for chunk in document_data['chunks']])

    prompt_template = """En base a las resoluciones o el apartado donde se resuelve a los miembros del comité técnico, lista a cada miembro y su función tanto en la empresa como en el comité.
    -Contexto: Se necesita extraer información de los miembros del comité técnico del proceso para alimentar las bases de datos.
    -Instrucción: Extrae la información de todos los miembros del comité técnico y sus cargos tanto para con el comité como para con la empresa
    -Formato: Muestra la información solicitada en formato JSON, siguiendo el orden de Miembro del comité, Cargo en la empresa, Cargo en la comisión
    -Restricciones: Si no encuentras la información en el texto no la pongas o no te inventes nombres, y no pongas el nombre de la persona que firmó el documento"""

    prompt = f"Context:\n{context}\n\n{prompt_template}"
    response_text = generate_ai_response(prompt)

    # Store the AI-generated response
    responses_store[filename] = response_text
    save_json_store(RESPONSES_STORE, responses_store)

    return jsonify({'answer': response_text})


@app.route('/responses', methods=['GET'])
def get_saved_response():
    filename = request.args.get('filename', '')
    if filename in responses_store:
        return jsonify({'answer': responses_store[filename]})
    return jsonify({'answer': 'No saved response found.'})


@app.route('/export_csv', methods=['GET'])
def export_csv():
    """Exports all stored responses in JSON format to a CSV file with proper UTF-8 encoding."""
    csv_output = []

    # Cargar los datos desde responses_store.json
    responses_data = load_json_store(RESPONSES_STORE)

    if not responses_data:
        print("No hay datos en responses_store.json")
        return Response("No hay datos para exportar", mimetype="text/plain")

    # Recorrer cada documento y extraer información
    for filename, response_text in responses_data.items():
        doc_name = filename.replace('.pdf', '').replace('.PDF', '')  # Remover extensión

        # Intentar limpiar el JSON del texto
        cleaned_text = re.sub(r'```json|```', '', response_text).strip()

        try:
            # Intentar cargar como JSON
            members_data = json.loads(cleaned_text)

            if not isinstance(members_data, list):
                print(f"Formato incorrecto en {filename}, se esperaba una lista.")
                continue

            # Extraer cada miembro y agregarlo al CSV
            for member in members_data:
                row = [
                    doc_name,
                    member.get("Miembro del comité", "N/A"),
                    member.get("Cargo en la empresa", "N/A"),
                    member.get("Cargo en la comisión", "N/A")
                ]
                csv_output.append(row)
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON en {filename}: {e}")

    # Si no hay datos extraídos, informar
    if not csv_output:
        print("No se extrajeron datos válidos del JSON.")
        return Response("No hay datos válidos para exportar", mimetype="text/plain")

    # Crear CSV en memoria con UTF-8-SIG
    def generate():
        output = io.StringIO()
        writer = csv.writer(output, delimiter='\t')
        writer.writerow(["Nombre del documento", "Nombre del miembro", "Cargo en la empresa", "Cargo en la comisión"])  # Cabecera
        for row in csv_output:
            writer.writerow(row)
        output.seek(0)  # Reiniciar puntero de lectura
        yield output.read()
        output.close()

    return Response(generate(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=export.csv"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
