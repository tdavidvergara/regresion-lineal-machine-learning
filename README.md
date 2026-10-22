#Tarea 1: Problemas de Regresión — Aprendizaje Automático

**Universidad Autónoma de Occidente**  
**Facultad de Ingeniería**  
**Asignatura:** Aprendizaje Automático  
**Profesor:** Juan Camilo Giraldo Londoño  
**Integrantes:** 

=======

David Alberto Vergara Tabares

Sebastián Giraldo

Violeta España

Murillo

## Descripción del proyecto

Este proyecto tiene como objetivo analizar un conjunto de datos (*dataset*) para resolver un problema de regresión mediante la implementación y evaluación de diferentes métodos de regresión lineal.

A partir del análisis exploratorio de los datos, se identificarán las variables predictoras más adecuadas para estimar una variable objetivo. Posteriormente, se construirán modelos de regresión utilizando diferentes enfoques matemáticos y computacionales, con el propósito de comparar su rendimiento y capacidad de generalización sobre datos no utilizados durante el entrenamiento.

El proyecto se desarrollará en un notebook de **Google Colab**, utilizando Python y bibliotecas especializadas en análisis de datos y aprendizaje automático.

##  Objetivos

### Objetivo general

Implementar, evaluar y comparar diferentes métodos de regresión lineal para resolver un problema de predicción, analizando su desempeño mediante métricas estadísticas y de error.

### Objetivos específicos

- Realizar un análisis exploratorio de datos (EDA) para comprender las características del conjunto de datos.
- Identificar las variables predictoras más relevantes para el problema de regresión.
- Detectar y analizar posibles valores atípicos (*outliers*).
- Dividir el conjunto de datos en subconjuntos de entrenamiento y validación.
- Estimar los parámetros del modelo mediante la ecuación normal, Scikit-learn y el descenso del gradiente.
- Implementar modelos con regularización Lasso y Ridge.
- Evaluar y comparar los modelos utilizando métricas de desempeño.
- Analizar el posible sobreajuste (*overfitting*) y la capacidad de generalización de los modelos.

##  Metodología

El desarrollo del proyecto seguirá las siguientes etapas:

### 1. Selección y comprensión del dataset

Se seleccionará uno de los conjuntos de datos propuestos para la actividad. Se identificarán sus variables, tipos de datos, dimensiones y características principales.

### 2. Análisis exploratorio de datos (EDA)

Se estudiará la distribución de las variables y sus relaciones mediante estadísticas descriptivas y visualizaciones.

Se analizarán aspectos como:

- Valores faltantes y registros duplicados.
- Distribución de las variables.
- Correlación entre las variables predictoras y la variable objetivo.
- Posibles valores atípicos.
- Variables potencialmente útiles para la regresión.

### 3. Preparación de los datos

Se realizarán las transformaciones necesarias para preparar los datos para el entrenamiento, incluyendo la selección de variables, el tratamiento de valores faltantes cuando corresponda y el escalado de características cuando sea necesario.

El conjunto de datos se dividirá en entrenamiento y validación para evaluar el rendimiento del modelo sobre observaciones no utilizadas durante el ajuste.

### 4. Implementación de los modelos de regresión

Se estimarán los parámetros de la regresión lineal mediante los siguientes métodos:

| Método | Descripción |
|---|---|
| Ecuación normal | Obtención analítica de los coeficientes de regresión mediante mínimos cuadrados ordinarios. |
| Scikit-learn | Implementación de regresión lineal utilizando la clase `LinearRegression`. |
| Lasso | Regresión lineal con regularización L1 para penalizar los coeficientes y favorecer la selección de variables. |
| Ridge | Regresión lineal con regularización L2 para reducir la magnitud de los coeficientes. |
| Descenso del gradiente | Método iterativo para minimizar la función de costo y encontrar los parámetros del modelo. |

### 5. Evaluación y comparación

Los modelos se evaluarán sobre los conjuntos de entrenamiento y validación, comparando sus resultados mediante métricas estadísticas y de error.

Se analizarán las diferencias entre los métodos, su capacidad predictiva y su comportamiento frente a posibles problemas de generalización.

## Métricas de evaluación

Se utilizarán las siguientes métricas para medir el rendimiento de los modelos:

| Métrica | Descripción |
|---|---|
| Coeficientes de regresión | Representan la relación estimada entre las variables predictoras y la variable objetivo. |
| \(R^2\) Score | Mide la proporción de variabilidad de la variable objetivo explicada por el modelo. |
| MSE | Calcula el promedio de los errores de predicción al cuadrado. |
| RMSE | Corresponde a la raíz cuadrada del MSE y expresa el error en las unidades de la variable objetivo. |

Las métricas se calcularán tanto para entrenamiento como para validación, con el fin de identificar diferencias en el rendimiento y posibles indicios de sobreajuste.

## Preguntas de análisis

Como parte de las conclusiones del proyecto, se responderán las siguientes preguntas:

1. ¿Es posible que se presente el fenómeno de sobreajuste (*overfitting*) en la regresión lineal? ¿Se presentó en el modelo entrenado?
2. ¿El conjunto de datos seleccionado presenta valores atípicos (*outliers*)? ¿Cómo se identificaron?
3. ¿Es necesario eliminar los valores atípicos? ¿Por qué sí o por qué no?
4. ¿Qué es el escalado (*normalization*) y en qué situaciones es necesario?
5. ¿En qué escenarios es preferible utilizar el descenso del gradiente en lugar de la regresión ordinaria por mínimos cuadrados mediante la ecuación normal, y por qué?

Las respuestas se fundamentarán en los resultados experimentales, las visualizaciones y los conceptos teóricos estudiados.


