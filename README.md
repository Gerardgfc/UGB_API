# UGB_API: Detección de Fraudes Financieros con Machine Learning

## Descripción
UGB_API es una aplicación de programación de interfaces (API) diseñada para detectar transacciones financieras fraudulentas. Emplea un modelo de `GradientBoostingClassifier` entrenado en un conjunto de datos financieros para identificar patrones anómalos que puedan indicar actividad fraudulenta. La API recibe como entrada un archivo CSV con datos estructurados de transacciones y devuelve un archivo CSV con las predicciones correspondientes, clasificando cada transacción como "fraude" o "no fraude".

## Arquitectura y Funcionamiento
- **Preprocesamiento de Datos**: El modelo subyacente ha sido entrenado utilizando un conjunto de datos cuidadosamente seleccionado y preprocesado. Este proceso incluye la limpieza de datos, la transformación de variables y la ingeniería de características para optimizar el rendimiento del modelo.
- **Modelo de Machine Learning**: Se ha utilizado un modelo de `GradientBoostingClassifier` debido a su capacidad para capturar interacciones complejas entre las variables y su alto rendimiento en problemas de clasificación.
- **API RESTful**: La API expone un endpoint RESTful que permite a los clientes enviar solicitudes HTTP POST con los datos de entrada y recibir las predicciones en formato CSV.

## Instalación y Uso

### Clonación del Repositorio:
```bash
git clone https://github.com/Gerardgfc/UGB_API.git
```
## Creación de un Entorno Virtual:

```bash
python -m venv venv
source venv/bin/activate
```
##Instalación de Dependencias:

````bash
pip install -r requirements.txt
````
