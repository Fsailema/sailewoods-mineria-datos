
# ============================================
# PROYECTO: Minería de Datos - SaileWood's
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# --------------------------------------------
# 1. Cargar datos
# --------------------------------------------
data = pd.read_excel("data/dataset_sailewoods.xlsx")

print("Columnas del dataset:")
print(data.columns)

print("\nPrimeros datos:")
print(data.head())

# --------------------------------------------
# 2. Preprocesamiento
# --------------------------------------------

# Convertir nombres a minúsculas
data.columns = data.columns.str.lower()

# Eliminar nulos
data = data.dropna()

# --------------------------------------------
# 3. Análisis exploratorio
# --------------------------------------------

# Ventas por producto
ventas_producto = data.groupby('producto')['total_venta'].sum()

print("\nVentas por producto:")
print(ventas_producto)

# Crear carpeta
os.makedirs("resultados/graficos", exist_ok=True)

# Gráfico
plt.figure()
ventas_producto.plot(kind='bar')
plt.title("Ventas por producto")
plt.xlabel("Producto")
plt.ylabel("Total ventas")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("resultados/graficos.png")
plt.close()

# --------------------------------------------
# 4. Modelado (Clustering)
# --------------------------------------------

# Variables para clustering
X = data[['precio_unitario', 'cantidad']]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

data['cluster'] = kmeans.labels_

print("\nDatos con clusters:")
print(data.head())

# --------------------------------------------
# 5. Guardar resultados
# --------------------------------------------

os.makedirs("resultados", exist_ok=True)
data.to_csv("resultados/datos_clasificados.csv", index=False)

print("\nProceso completado correctamente.")
