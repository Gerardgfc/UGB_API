from flask import Flask, request, jsonify, send_file, render_template
import joblib
import pandas as pd
import os
import sys  # Asegúrate de importar sys
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
columnas_modelo = [...]  # Mantén tu lista de columnas aquí

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

    # Generar el CSV en memoria
    output = BytesIO()
    resultados_df.to_csv(output, index=False)
    output.seek(0)

    # Usar un nombre único para el archivo
    output_file_name = 'resultado.csv'

    return send_file(output, mimetype='text/csv', as_attachment=True, download_name=output_file_name)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv('PORT', default=5000))
