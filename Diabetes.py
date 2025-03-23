import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Habilitamos IterativeImputer (característica experimental)
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
# Configuración de estilo
sns.set(style="whitegrid", font_scale=1.2)

# Cargamos el dataset
file_path = './content/diabetes.csv'
df = pd.read_csv(file_path, sep=';')  # Se usa el separador ';' para una lectura correcta

# Eliminamos duplicados y reemplazamos -999 por NaN
df = df.drop_duplicates()
df.replace(-999, np.nan, inplace=True)

# Calculamos la proporción de individuos con valores nulos (por filas)
df_nulos = df[df.isnull().any(axis=1)]
total_individuos = len(df)
individuos_con_nulos = len(df_nulos)
proporcion = (individuos_con_nulos / total_individuos) * 100
print(f"Proporción de individuos con valores perdidos (tras eliminar duplicados): {proporcion:.2f}%")

# --- Visualizaciones de valores nulos ---
# 1. Matriz de correlaciones para el DataFrame completo
corr_full = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_full, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5)
plt.title('Matriz de correlaciones del DataFrame completo')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Matriz de correlaciones para filas con valores nulos (si existen)
if not df_nulos.empty:
    corr_nulos = df_nulos.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_nulos, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5)
    plt.title('Matriz de correlaciones en filas con valores nulos')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("No hay filas con valores nulos para calcular la correlación.")

# 3. Gráfico de barras: Número de valores nulos por columna
missing_counts = df.isnull().sum()
cols_con_nulos = missing_counts[missing_counts > 0]
plt.figure(figsize=(12, 8))
sns.barplot(x=cols_con_nulos.index, y=cols_con_nulos.values, palette="viridis")
plt.title("Número de valores nulos por columna")
plt.xlabel("Columnas")
plt.ylabel("Número de nulos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Mapa de calor: Patrón de valores nulos en el DataFrame
plt.figure(figsize=(16, 12))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Patrón de valores nulos en el DataFrame")
plt.xlabel("Columnas")
plt.ylabel("Filas")
plt.tight_layout()
plt.show()

# 5. Histogramas individuales para cada columna con valores nulos
if not cols_con_nulos.empty:
    num_plots = len(cols_con_nulos)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, num_plots * 4))
    if num_plots == 1:
        axes = [axes]
    for ax, col in zip(axes, cols_con_nulos.index):
        datos = df[col].dropna()
        sns.histplot(datos, kde=True, ax=ax, color="blue")
        ax.set_title(f"Distribución de {col} (sin nulos)")
        ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()
else:
    print("No hay columnas con valores nulos para graficar.")

# --- Imputación con IterativeImputer ---
# Creamos el imputador con una semilla para reproducibilidad
imputer = IterativeImputer(random_state=89)

df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Redondeamos las columnas que deben ser binarias
binary_columns = ['CholCheck', 'HeartDiseaseorAttack', 'NoDocbcCost']
for col in binary_columns:
    if col in df_imputed.columns:
        df_imputed[col] = np.round(df_imputed[col]).astype(int)

print("Primeras filas del dataset imputado con variables binarias redondeadas:")
print(df_imputed.head())


# --- Detección de outliers con IsolationForest ---
# Definimos la proporción de datos que esperamos sean outliers.
# La elección del parámetro 'contamination' depende de tu caso concreto.
# Se define la contaminación (porcentaje de outliers esperado)
# Supongamos que ya tienes tu dataset imputado en df_imputed
# Ejemplo: df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Filtramos las observaciones donde Diabetes_binary es 0
df_diab0 = df_imputed[df_imputed['Diabetes_binary'] == 0].copy()

# Es opcional: si deseas excluir la variable binaria de la imputación de outliers, puedes quitarla:
features = df_diab0.drop(columns=['Diabetes_binary','Sex','NoDocbcCost'])

# Configuramos IsolationForest para detectar outliers en el subconjunto
iso_forest = IsolationForest(contamination=0.005, random_state=42)
outlier_labels = iso_forest.fit_predict(features)  # -1: outlier, 1: inlier

# Añadimos la etiqueta de outlier al DataFrame filtrado
df_diab0['outlier'] = outlier_labels

# Obtenemos los índices de los outliers en el subconjunto de Diabetes_binary = 0
outlier_indices = df_diab0[df_diab0['outlier'] == -1].index

# Filtramos el dataset df_imputed para eliminar esos registros
df_filtrado = df_imputed.drop(index=outlier_indices)

print("Dataset df_imputed sin los outliers (casos con Diabetes_binary = 0):")
print(df_filtrado.head())
print("Dimensiones del dataset filtrado:", df_filtrado.shape)

# Visualizamos el pairplot del dataset filtrado, diferenciando por Diabetes_binary
sns.pairplot(data=df_filtrado, hue='Diabetes_binary', diag_kind='kde')
plt.suptitle('Pairplot del dataset filtrado', y=1.02)
plt.show()

# Calculamos la matriz de correlaciones del dataset filtrado
corr_filtrado = df_filtrado.corr()

# Mostramos la matriz de correlaciones con un heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr_filtrado, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5)
plt.title('Matriz de correlaciones del dataset filtrado')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()