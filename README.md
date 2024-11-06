![DataFOrge](https://drive.google.com/u/0/drive-viewer/AKGpihawFnYHp64II-wv4GtYB-DhJXG-BJRDuKo5Tcjuzww9PWJyo9ol-tt_Tmz6eXdoqMEqrbUoNUlOj_3mMJN1nbJ96g97AbjeJQ=s2560)

# UGB_API: Detección de Fraudes Financieros con Machine Learning

## Descripción
UGB_API es una aplicación de programación de interfaces (API) diseñada para detectar transacciones financieras fraudulentas. Emplea un modelo de `GradientBoostingClassifier` entrenado en un conjunto de datos financieros para identificar patrones anómalos que puedan indicar actividad fraudulenta. La API recibe como entrada un archivo CSV con datos estructurados de transacciones y devuelve un archivo CSV con las predicciones correspondientes, clasificando cada transacción como "fraude" o "no fraude".

## Arquitectura y Funcionamiento
- **Preprocesamiento de Datos**: El modelo subyacente ha sido entrenado utilizando un conjunto de datos cuidadosamente seleccionado y preprocesado. Este proceso incluye la limpieza de datos, la transformación de variables y la ingeniería de características para optimizar el rendimiento del modelo.
- **Modelo de Machine Learning**: Se ha utilizado un modelo de `GradientBoostingClassifier` debido a su capacidad para capturar interacciones complejas entre las variables y su alto rendimiento en problemas de clasificación.
- **API RESTful**: La API expone un endpoint RESTful que permite a los clientes enviar solicitudes HTTP POST con los datos de entrada y recibir las predicciones en formato CSV.

## Instalación y Uso

### 1.Clonación del Repositorio:
```bash
git clone https://github.com/Gerardgfc/UGB_API.git
```
## 2.Creación de un Entorno Virtual:

```bash
python -m venv venv
source venv/bin/activate
```
## 3.Instalación de Dependencias:

````bash
pip install -r requirements.txt
````
## 4.Ejecución de la API:

````bash
python main.py
````
## 5.Realización de una Petición:

````bash
curl -X POST -F file=@tu_archivo.csv https://ugb-api-tests.onrender.com/
````

Donde <code>tu_archivo.csv</code> es el archivo CSV con los datos de las transacciones a evaluar.

## Estructura del Proyecto

- **data**: Contiene los datos utilizados para el entrenamiento y la evaluación del modelo.
- **models**: Almacena los modelos entrenados.
- **notebooks**: Incluye los notebooks de Jupyter utilizados para el preprocesamiento, entrenamiento y evaluación del modelo.
- **src**: Contiene el código fuente de la API.

## Formatos de Datos

- **Entrada**: Archivo CSV con las siguientes columnas: `['ID', 'income', 'name_email_similarity', 'current_address_months_count', 
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
    'device_os_other', 'device_os_windows', 'device_os_x11']`.
- **Salida**: Archivo CSV con las mismas columnas de entrada y una columna adicional llamada "predicciones" con los valores "fraude" o "no fraude".

## Limitaciones

- **Tamaño de Archivo**: La API está limitada a procesar archivos CSV con un máximo de 999.999 filas.
- **Modelo**: El modelo ha sido entrenado en un conjunto de datos específico y puede no generalizar perfectamente a todos los escenarios de fraude.

## Subjerencias

Se valoran las sugerencias y comentarios a través de LinkedIn: [Gerardo Carrizo](https://www.linkedin.com/in/gerardo-carrizo/)

## Front-end

La API alimenta un front-end desarrollado en [Js] y alojado en:

[Web de dataforge](https://gerardgfc.github.io/dataforge/)

El código fuente del front-end se encuentra en el repositorio:

[Repositorio dataforge](https://github.com/Gerardgfc/dataforge)

## Autor

[Gerardo Carrizo](https://www.linkedin.com/in/gerardo-carrizo/)  

