import os
import re
import csv
import io
import json
import datetime
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
    # Get the current page from query parameters, default to 1
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '', type=str)  # Get search query
    per_page = 10  # Number of files to display per page
    files = list(embeddings_store.keys())

    # Filter files based on search query
    if search_query:
        files = [file for file in files if search_query.lower() in file.lower()]

    # Calculate total pages
    total_files = len(files)
    total_pages = (total_files + per_page - 1) // per_page  # Ceiling division

    # Get the files for the current page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_files = files[start:end]

    return render_template("index.html", files=paginated_files, total_pages=total_pages, current_page=page, search_query=search_query)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('file')
    for file in files:
        if file.filename == '' or not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            continue
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text_chunks = extract_text_chunks(file_path)
        embeddings = generate_text_embeddings(text_chunks)
        if embeddings:
            embeddings_store[filename] = {'chunks': embeddings}
            save_json_store(JSON_STORE, embeddings_store)
    return redirect(url_for('index'))


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
    -Restricciones: Si no encuentras la información en el texto no la pongas o no te inventes nombres, y no pongas el nombre de la persona que firmó el documento, además si no encuentras el cargo dentro de la comisión, repite en ese campo el cargo en la empresa pero no puede decir simplemente miembro, y si no encuentras el cargo en la empresa repite el cargo en el comité"""

    prompt = f"Context:\n{context}\n\n{prompt_template}"
    response_text = generate_ai_response(prompt)
    responses_store[filename] = response_text
    save_json_store(RESPONSES_STORE, responses_store)
    return jsonify({'answer': response_text})


@app.route('/ask_all', methods=['POST'])
def ask_all():
    responses = {}
    for filename, document_data in embeddings_store.items():
        context = "\n".join([chunk[0] for chunk in document_data['chunks']])
        prompt_template = """En base a las resoluciones o el apartado donde se resuelve a los miembros del comité técnico, lista a cada miembro y su función tanto en la empresa como en el comité.
            -Contexto: Se necesita extraer información de los miembros del comité técnico del proceso para alimentar las bases de datos.
            -Instrucción: Extrae la información de todos los miembros del comité técnico y sus cargos tanto para con el comité como para con la empresa
            -Formato: Muestra la información solicitada en formato JSON, siguiendo el orden de Miembro del comité, Cargo en la empresa, Cargo en la comisión
            -Restricciones: Si no encuentras la información en el texto no la pongas o no te inventes nombres, y no pongas el nombre de la persona que firmó el documento, además si no encuentras el cargo dentro de la comisión, repite en ese campo el cargo en la empresa pero no puede decir simplemente miembro, y si no encuentras el cargo en la empresa repite el cargo en el comité"""

        prompt = f"Context:\n{context}\n\n{prompt_template}"
        response_text = generate_ai_response(prompt)
        responses_store[filename] = response_text
        save_json_store(RESPONSES_STORE, responses_store)
        responses[filename] = response_text
    return jsonify(responses)


@app.route('/responses', methods=['GET'])
def get_saved_response():
    filename = request.args.get('filename', '')
    if filename in responses_store:
        return jsonify({'answer': responses_store[filename]})
    return jsonify({'answer': 'No saved response found.'})


@app.route('/export_csv', methods=['GET'])
def export_csv():
    """Exports all stored responses in JSON format to a properly encoded CSV file with a timestamped filename."""
    csv_output = []

    # Load stored responses from JSON file
    responses_data = load_json_store(RESPONSES_STORE)

    if not responses_data:
        print("No hay datos en responses_store.json")
        return Response("No hay datos para exportar", mimetype="text/plain")

    # Process each document and extract relevant data
    for filename, response_text in responses_data.items():
        doc_name = filename.replace('.pdf', '').replace('.PDF', '')  # Remove PDF extension

        # Remove markdown ```json ``` blocks
        cleaned_text = re.sub(r'```json|```', '', response_text).strip()

        try:
            # Parse JSON
            members_data = json.loads(cleaned_text)

            if not isinstance(members_data, list):
                print(f"Formato incorrecto en {filename}, se esperaba una lista.")
                continue

            # Extract data and ensure it's clean
            for member in members_data:
                row = [
                    doc_name,
                    (member.get("Miembro del comité") or "").strip(),
                    (member.get("Cargo en la empresa") or "").strip(),
                    (member.get("Cargo en la comisión") or "").strip()
                ]
                csv_output.append(row)
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON en {filename}: {e}")

    # Ensure there is data to write
    if not csv_output:
        print("No se extrajeron datos válidos del JSON.")
        return Response("No hay datos válidos para exportar", mimetype="text/plain")

    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}-CELEC-COM.csv"

    # Create CSV with UTF-8 BOM (Byte Order Mark) for proper encoding
    def generate():
        output = io.StringIO()
        output.write('\ufeff')  # Add UTF-8 BOM for Excel compatibility
        writer = csv.writer(output, delimiter='\t')

        # Write headers
        writer.writerow(["Código del proceso", "Nombre del miembro", "Cargo en la empresa", "Cargo en la comisión"])
        for row in csv_output:
            writer.writerow(row)

        output.seek(0)  # Reset pointer to the beginning
        yield output.read()
        output.close()

    return Response(generate(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route('/delete', methods=['POST'])
def delete_file():
    data = request.json
    filename = data.get('filename', '')

    if filename in embeddings_store:
        del embeddings_store[filename]
        save_json_store(JSON_STORE, embeddings_store)

    if filename in responses_store:
        del responses_store[filename]
        save_json_store(RESPONSES_STORE, responses_store)

    # Optionally, delete the actual file from the uploads directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return jsonify({'message': 'File and associated data deleted successfully.'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
