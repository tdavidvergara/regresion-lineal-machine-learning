# Estructura del proyecto

Tarea 1 de Aprendizaje Automático: regresión lineal sobre el dataset
[Daily Demand Forecasting Orders](https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders) (UCI, 60 días, 12 variables predictoras).

## Archivos

| Archivo | Descripción |
|---|---|
| `EDA_Regresion_Demanda.ipynb` | Notebook principal: EDA, modelado y respuestas a las preguntas. |
| `dataset/Daily_Demand_Forecasting_Orders.csv` | Datos originales de UCI (separador `;`). Es el que usa el notebook. |
| `dataset/Dataset_Mod.csv` | Versión modificada del dataset (no se usa en el notebook). |
| `README.md` | Descripción general, metodología, métricas y preguntas de la tarea. |
| `pyproject.toml`, `uv.lock` | Dependencias del entorno (se instalan con `uv sync`). |

## Secciones del notebook

1. **EDA**
   - Carga y ficha técnica de los datos.
   - Estadísticas descriptivas.
   - Correlación (mapa de calor, correlación con el target y diagramas de dispersión).
   - Valores faltantes y duplicados.
   - Valores atípicos (IQR, Z-score y boxplots).
   - Fuga de datos: el target es exactamente `Order type A + B + C`.
   - Multicolinealidad (VIF).
   - Selección de variables y escalado.
2. **División** entrenamiento/validación (80 % / 20 %, `random_state=42`).
3. **Modelado**
   - Tres conjuntos de variables: completo (con fuga), sin fuga (9) y reducido (4).
   - Ecuación normal, descenso del gradiente, y scikit-learn (`LinearRegression`, Ridge y Lasso con validación cruzada para `alpha`).
   - Métricas: R², MSE y RMSE en entrenamiento y validación.
4. **Análisis de sobreajuste** (residuos de validación y validación cruzada) y **efecto de los outliers**.
5. **Respuestas** a las cinco preguntas de la guía.

## Notas

- El escalador (`StandardScaler`) se ajusta solo con entrenamiento para evitar filtración de datos.
- En Google Colab hay que subir la carpeta `dataset/` (o montar Drive) porque el notebook lee el CSV con ruta relativa.
