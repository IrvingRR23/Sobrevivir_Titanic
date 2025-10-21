 
#  Análisis del Titanic con Árbol de Decisión

Este proyecto implementa un modelo de **árbol de decisión** para predecir la supervivencia de los pasajeros del Titanic, utilizando el conjunto de datos `DataSet_Titanic.csv`.  
El código entrena el modelo, evalúa su precisión y visualiza tanto la **matriz de confusión** como el **árbol de decisión** y la **importancia de los atributos**.

---

##  Características principales
- Lectura automática del dataset desde la carpeta del proyecto.
- Entrenamiento de un modelo con `DecisionTreeClassifier` (profundidad máxima = 5).
- Cálculo del porcentaje de **precisión** sobre los datos.
- Visualización de:
  - Matriz de confusión normalizada.
  - Árbol de decisión coloreado.
  - Importancia de cada atributo (gráfica con `seaborn`).

---

##  Tecnologías utilizadas
- **Python 3**
- **pandas** – manejo de datos  
- **scikit-learn** – modelo de árbol de decisión y métricas  
- **matplotlib** y **seaborn** – visualización de gráficos  


