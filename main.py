from flask import Flask, request, jsonify, send_file
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
    'ID', 'income', 'name_email_similarity', 'current_address_months_count', 
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
    try:
        data_df_normalizado = escalador.fit_transform(data_df)
        return pd.DataFrame(data_df_normalizado, columns=data_df.columns)
    except ValueError as e:
        raise ValueError(f"Error en la conversión de datos: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No hay parte de archivo en la solicitud'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Archivo no cargado'}), 400
    
    try:
        if file.filename.endswith('.csv'):
            data_df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data_df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Formato de archivo no soportado. Use CSV o Excel.'}), 400
    except Exception as e:
        return jsonify({'error': f'Error al leer el archivo: {str(e)}'}), 400

    data_df_preparado, error = preparar_dataframe(data_df)
    if error:
        return jsonify({'error': error}), 400

    id_column = data_df_preparado['ID']
    data_df_preparado = data_df_preparado.drop(columns=['ID'])

    try:
        data_df_preprocesado = preprocesar_datos(data_df_preparado)
    except ValueError as ve:
        return jsonify({'error': f'Error en el preprocesamiento de datos: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error en el preprocesamiento de datos: {str(e)}'}), 400

    try:
        prediccion = modelo.predict(data_df_preprocesado)
        prediccion_texto = ['fraude' if p == 1 else 'no fraude' for p in prediccion]

        resultados_df = pd.DataFrame(data={
            'ID': id_column.values,
            'predicciones': prediccion_texto
        })

        output = BytesIO()

        # Obtener el nombre original del archivo y construir el nombre del archivo de salida
        base_name = file.filename.rsplit('.', 1)[0]
        if file.filename.endswith('.csv'):
            resultados_df.to_csv(output, index=False)
            mimetype = 'text/csv'
            download_name = f'resultado_{base_name}.csv'
        elif file.filename.endswith('.xlsx'):
            resultados_df.to_excel(output, index=False, engine='openpyxl')
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            download_name = f'resultado_{base_name}.xlsx'
        
        output.seek(0)

        return send_file(output, mimetype=mimetype, as_attachment=True, download_name=download_name)
    except Exception as e:
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
