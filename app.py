from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Ruta al archivo del modelo
modelo_path = os.path.join(os.getcwd(), 'modelo_fraudev2.pkl')

# Cargar el modelo si existe
if os.path.exists(modelo_path):
    modelo = joblib.load(modelo_path)
else:
    print("Archivo de modelo no encontrado")
    sys.exit(1)

# Inicializar el escalador
escalador = StandardScaler()

# Definir las columnas necesarias
columnas_modelo = [
    'income', 'name_email_similarity', 'current_address_months_count', 
    'customer_age', 'days_since_request', 'bank_branch_count_8w', 
    'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'email_is_free', 
    'phone_home_valid', 'phone_mobile_valid', 'bank_months_count', 
    'has_other_cards', 'proposed_credit_limit', 'foreign_request', 
    'session_length_in_minutes', 'keep_alive_session', 
    'device_distinct_emails_8w', 'payment_type_AA', 'payment_type_AB', 
    'payment_type_AC', 'payment_type_AD', 'payment_type_AE', 
    'employment_status_CA', 'employment_status_CB', 'employment_status_CC', 
    'employment_status_CD', 'employment_status_CE', 'employment_status_CF', 
    'employment_status_CG', 'housing_status_BA', 'housing_status_BB', 
    'housing_status_BC', 'housing_status_BD', 'housing_status_BE', 
    'housing_status_BF', 'housing_status_BG', 'source_INTERNET', 
    'source_TELEAPP', 'device_os_linux', 'device_os_macintosh', 
    'device_os_other', 'device_os_windows', 'device_os_x11'
]

def preparar_dataframe(df):
    # Verificar si las columnas requeridas están en el DataFrame
    faltantes = [col for col in columnas_modelo if col not in df.columns]
    if faltantes:
        return None, f"Faltan las siguientes columnas: {', '.join(faltantes)}"
    
    df_filtrado = df[columnas_modelo].copy()
    return df_filtrado, None

def preprocesar_datos(data_df):
    data_df_normalizado = escalador.fit_transform(data_df)
    return pd.DataFrame(data_df_normalizado, columns=data_df.columns)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predicción de Fraude</title>
        <style>
            body {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                text-align: center;
                background-color: rgb(13, 14, 26);
                color: white;
            }
            .file-label {
                border: 2px solid white;
                padding: 10px;
                border-radius: 5px;
                background-color: transparent;
                color: white;
                cursor: pointer;
            }
            .selected-file {
                margin: 20px 0;
                color: white;
            }
            #loading, #result {
                color: white;
                font-size: 22px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Subir archivo para predicción de fraude</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <label class="file-label" for="file-input">Seleccionar archivo</label>
            <input type="file" id="file-input" name="file" required onchange="updateFileName()" style="display: none;">
            <div class="selected-file" id="selected-file">Archivos seleccionados: Ninguno</div>
            <button type="submit">Enviar</button>
        </form>
        <p>Recordar que debe tener las columnas adecuadas.</p>
        <div id="loading" style="display: none;">Cargando<span id="dots">...</span></div>
        <div id="result"></div>

        <script>
            function updateFileName() {
                const input = document.getElementById('file-input');
                const label = document.getElementById('selected-file');
                const fileName = input.files.length > 0 ? input.files[0].name : 'Ninguno';
                label.textContent = `Archivos seleccionados: ${fileName}`;
            }

            document.getElementById('uploadForm').onsubmit = async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                const loadingMessage = document.getElementById('loading');
                const dots = document.getElementById('dots');

                loadingMessage.style.display = 'block';

                let dotCount = 0;
                const dotAnimation = setInterval(() => {
                    dotCount = (dotCount + 1) % 4;
                    dots.textContent = '.'.repeat(dotCount);
                }, 500);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    clearInterval(dotAnimation);
                    loadingMessage.style.display = 'none';

                    if (data.output_file) {
                        const downloadLink = `<a href="/download/${data.output_file}" download>Descargar resultados</a>`;
                        document.getElementById('result').innerHTML = downloadLink;
                    } else {
                        document.getElementById('result').innerText = data.error;
                    }
                } catch (error) {
                    clearInterval(dotAnimation);
                    loadingMessage.style.display = 'none';
                    document.getElementById('result').innerText = 'Error en la conexión. Inténtalo de nuevo.';
                    console.error(error);
                }
            };
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No hay parte de archivo en la solicitud'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo no cargado'}), 400
    
    try:
        data_df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error al leer el archivo CSV: {str(e)}'}), 400

    data_df_preparado, error = preparar_dataframe(data_df)
    if error:
        return jsonify({'error': error}), 400

    data_df_preprocesado = preprocesar_datos(data_df_preparado)

    prediccion = modelo.predict(data_df_preprocesado)
    resultados_df = pd.DataFrame(data={'predicciones': prediccion})

    output_folder = 'resultados'
    os.makedirs(output_folder, exist_ok=True)

    original_filename = os.path.splitext(file.filename)[0]
    output_file_name = f"resultado_{original_filename}.csv"
    output_file_path = os.path.join(output_folder, output_file_name)

    resultados_df.to_csv(output_file_path, index=False)

    return jsonify({'message': 'Predicciones guardadas en CSV', 'output_file': output_file_name})

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('resultados', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
