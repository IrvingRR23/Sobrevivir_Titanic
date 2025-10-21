import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn import tree
import os

# Leer dataset
ruta_dataset = os.path.join(os.path.dirname(__file__), "DataSet_Titanic.csv")
df = pd.read_csv(ruta_dataset)

# Separar variables
X = df.drop("Sobreviviente", axis=1)
y = df["Sobreviviente"]

# Crear modelo
arbol = DecisionTreeClassifier(max_depth=5, random_state=42)
arbol.fit(X, y)

# Predicciones
pred_y = arbol.predict(X)

# Precisión
print("Precisión: {:.2%}".format(accuracy_score(y, pred_y)))  

# Matriz de confusión con colores
ConfusionMatrixDisplay.from_estimator(
    arbol, X, y, normalize="true", cmap="Blues"
)
plt.title("Matriz de Confusión (Normalizada)")
plt.show()

# Árbol de decisión con fondo claro
plt.figure(figsize=(12, 8))
tree.plot_tree(
    arbol,
    filled=True,                 
    feature_names=X.columns,
    class_names=["No", "Sí"]    
)
plt.title("Árbol de Decisión - Titanic")
plt.show()

# Importancia de atributos con paleta
importancias = arbol.feature_importances_
sns.barplot(x=X.columns, y=importancias, palette="viridis")
plt.title("Importancia de cada atributo")
plt.ylabel("Importancia")
plt.show()
