from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
import joblib
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app)

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
    faltantes = [col for col in columnas_modelo if col not in df.columns]
    if faltantes:
        return None, f"Faltan las siguientes columnas: {', '.join(faltantes)}"
    
    if len(df) > 999999:
        return None, "El archivo excede el número máximo de filas (999.999)."

    df_filtrado = df[columnas_modelo].copy()
    return df_filtrado, None

def preprocesar_datos(data_df):
    data_df_normalizado = escalador.fit_transform(data_df)
    return pd.DataFrame(data_df_normalizado, columns=data_df.columns)

@app.route('/')
def home():
    return render_template('index.html')

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

    try:
        data_df_preprocesado = preprocesar_datos(data_df_preparado)
    except Exception as e:
        return jsonify({'error': f'Error en el preprocesamiento de datos: {str(e)}'}), 400

    try:
        prediccion = modelo.predict(data_df_preprocesado)
        resultados_df = pd.DataFrame(data={'predicciones': prediccion})

        output = BytesIO()
        resultados_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='resultado.csv')
    except Exception as e:
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv('PORT', default=5000))
