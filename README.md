# Tarea 1: Problemas de Regresión — Aprendizaje Automático

**Universidad Autónoma de Occidente**  
**Facultad de Ingeniería**  
**Asignatura:** Aprendizaje Automático  
**Profesor:** Juan Camilo Giraldo Londoño  
**Integrantes:**

- David Alberto Vergara Tabares
- Sebastián Giraldo
- Violeta España
- Murillo

## Descripción del proyecto

Este proyecto analiza un conjunto de datos (*dataset*) para resolver un problema de regresión mediante la implementación y evaluación de diferentes métodos de regresión lineal.

A partir del análisis exploratorio de los datos se identifican las variables predictoras más adecuadas para estimar una variable objetivo. Luego se construyen modelos de regresión con distintos enfoques matemáticos y computacionales, y se compara su rendimiento y su capacidad de generalización sobre datos no usados en el entrenamiento.

El trabajo se desarrolla en un notebook de Python (`EDA_Regresion_Demanda.ipynb`), pensado para entregarse en **Google Colab**.

## Dataset

[**Daily Demand Forecasting Orders**](https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders) (UCI Machine Learning Repository).

- 60 registros (días) y 13 columnas numéricas, sin valores faltantes ni duplicados.
- **Variable objetivo:** `Target (Total orders)`, total de pedidos del día.
- **Predictoras:** semana del mes, día de la semana, pedidos urgentes y no urgentes, pedidos por tipo (A, B y C) y pedidos por sector (fiscal, control de tráfico y tres de banca).

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `EDA_Regresion_Demanda.ipynb` | Notebook principal: EDA, modelado y respuestas a las preguntas. |
| `dataset/Daily_Demand_Forecasting_Orders.csv` | Datos originales de UCI (separador `;`), usados por el notebook. |
| `dataset/Dataset_Mod.csv` | Versión modificada del dataset (no se usa en el notebook). |
| `ESTRUCTURA.md` | Detalle de las secciones del notebook. |
| `pyproject.toml`, `uv.lock` | Dependencias del entorno. |

## Cómo ejecutarlo

**En local** (requiere [uv](https://docs.astral.sh/uv/) y Python 3.12 o superior):

```bash
uv sync
uv run jupyter lab
```

Luego se abre `EDA_Regresion_Demanda.ipynb`. El notebook lee el CSV con ruta relativa (`dataset/...`), así que debe ejecutarse desde la raíz del repositorio.

**En Google Colab:** se sube el notebook y la carpeta `dataset/` (o se monta Drive) y se ajusta `RUTA_DATOS` si hace falta.

## Metodología

### 1. Análisis exploratorio de datos (EDA)

- Estadísticas descriptivas, mapa de correlación y diagramas de dispersión.
- Verificación de valores faltantes y duplicados.
- Detección de valores atípicos con el rango intercuartílico (IQR), el Z-score y boxplots.
- Detección de fuga de datos y análisis de multicolinealidad (VIF).
- Selección de variables y justificación del escalado.

### 2. Preparación de los datos

División aleatoria en entrenamiento (80 %) y validación (20 %) con `random_state=42`. Las variables se estandarizan con un `StandardScaler` **ajustado solo con el conjunto de entrenamiento**, para evitar filtración de datos.

### 3. Modelos de regresión

| Método | Descripción |
|---|---|
| Ecuación normal | Coeficientes de mínimos cuadrados ordinarios, calculados con la pseudoinversa. |
| Descenso del gradiente | Implementación propia que minimiza el error cuadrático medio de forma iterativa. |
| `LinearRegression` | Regresión lineal de scikit-learn. |
| Ridge | Regularización L2 (`RidgeCV`, con `alpha` elegido por validación cruzada). |
| Lasso | Regularización L1 (`LassoCV`, con `alpha` elegido por validación cruzada). |

Cada método se evalúa con tres conjuntos de variables:

| Conjunto | Variables |
|---|---|
| Completo (con fuga) | Las 12 predictoras. |
| Sin fuga | Las 9 predictoras sin `Order type A`, `B` ni `C`. |
| Reducido sin fuga | `Non-urgent order`, `Urgent order`, `Banking orders (2)` y `Day of the week`. |

### 4. Evaluación

Se calculan el R², el MSE y el RMSE sobre entrenamiento y validación, además de los coeficientes estimados. También se analiza el sobreajuste con los residuos de validación y con validación cruzada de 5 pliegues, y se mide el efecto de los outliers.

| Métrica | Descripción |
|---|---|
| Coeficientes de regresión | Relación estimada entre cada predictora y el objetivo. |
| R² | Proporción de la variabilidad del objetivo explicada por el modelo. |
| MSE | Promedio de los errores de predicción al cuadrado. |
| RMSE | Raíz del MSE, en las unidades del objetivo. |

## Fuga de datos (*data leakage*)

### Qué se encontró

La variable objetivo `Target (Total orders)` no es independiente de las predictoras: es **exactamente la suma de tres de ellas**:

```
Target (Total orders) = Order type A + Order type B + Order type C
```

El notebook lo verifica calculando la diferencia máxima entre el target y esa suma en los 60 días: da ≈ 6·10⁻¹⁴, es decir, cero salvo por el error de redondeo de los decimales.

### Cómo se detectó

Al entrenar el modelo con las 12 predictoras, el R² salió **1.0000** tanto en entrenamiento como en validación y el RMSE ≈ 0. Al revisar los coeficientes, los de `Order type A`, `B` y `C` eran los únicos distintos de cero, y los de las otras nueve variables salían en 0. Un ajuste perfecto en datos reales es una señal de alerta, y esa pista llevó a comprobar la relación anterior.

### Por qué es un problema

Hay fuga de datos cuando el modelo recibe como entrada información que contiene la respuesta. Aquí el modelo no aprende ninguna relación entre las variables y la demanda: solo suma tres columnas. Esto tiene varias consecuencias:

- **Las métricas no sirven.** Un R² de 1.0 no demuestra que el modelo generalice, solo que la respuesta ya estaba en la entrada.
- **No se puede comparar métodos ni hablar de sobreajuste.** Todos los métodos (ecuación normal, gradiente, Ridge y Lasso) dan casi lo mismo.
- **No es útil en la práctica.** Para predecir el total de pedidos de un día habría que conocer ya cuántos pedidos de cada tipo hubo, o sea, el dato que se quiere predecir.
- **Distorsiona el EDA.** Las correlaciones altas de `Order type B` (0.90) y `C` (0.80) con el target y parte de la multicolinealidad (VIF) se explican por esta suma, no por una relación real con la demanda.

### Cómo se trató

Se conservaron las dos versiones para que el efecto quede documentado:

| Conjunto | Variables | Resultado |
|---|---|---|
| **Completo (con fuga)** | Las 12 predictoras | R² ≈ 1.0 en entrenamiento y validación. Solo sirve como evidencia de la fuga. |
| **Sin fuga** | Las 9 predictoras restantes | R² ≈ 0.986 en entrenamiento y ≈ 0.78 en validación. Es el problema de regresión real. |
| **Reducido sin fuga** | 4 variables | R² ≈ 0.985 en entrenamiento y ≈ 0.78 en validación. |

Todas las conclusiones del proyecto (sobreajuste, outliers, Ridge y Lasso, comparación de métodos) se basan en los conjuntos **sin fuga**.

## Hallazgos principales

- **Fuga de datos:** el target es la suma de `Order type A + B + C` (ver la sección anterior). Por eso el análisis se hace sin esas tres columnas.
- **Outliers:** por IQR, 21 de las 60 filas tienen al menos un valor atípico (sobre todo en `Fiscal sector orders` y `Banking orders (1)`); por Z-score hay pocos. No se eliminaron: son días posibles del negocio y el dataset es pequeño.
- **Multicolinealidad:** `Non-urgent order` (VIF ≈ 56), `Order type C`, `Banking orders (1)` y `(2)` (VIF > 10).
- **Los cinco métodos coinciden** en los conjuntos sin fuga, lo que valida las implementaciones. Ridge y Lasso casi no cambian el resultado; Lasso deja `Non-urgent order` y `Urgent order` como las variables principales.
- **Desempeño (conjunto reducido sin fuga):** R² ≈ 0.985 en entrenamiento y ≈ 0.78 en validación; RMSE ≈ 9.8 y ≈ 47.3 pedidos.
- **Sobreajuste:** la brecha entre entrenamiento y validación se debe a un único día de validación con un error de +162 pedidos. Sin él, el R² de validación es 0.994, y en validación cruzada el promedio es ≈ 0.94. Con solo 12 días de validación, el resultado depende de la semilla de la división.

## Preguntas de análisis

El notebook responde, con base en los resultados experimentales:

1. ¿Es posible que se presente sobreajuste en la regresión lineal? ¿Se presentó en el modelo entrenado?
2. ¿El conjunto de datos presenta valores atípicos? ¿Cómo se identificaron?
3. ¿Es necesario eliminar los valores atípicos? ¿Por qué sí o por qué no?
4. ¿Qué es el escalado (*normalization*) y cuándo es necesario?
5. ¿En qué escenarios es preferible el descenso del gradiente a la ecuación normal y por qué?
