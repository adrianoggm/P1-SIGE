import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Habilitamos IterativeImputer (característica experimental)
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Importamos los filtros de ruido de imbalanced-learn
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
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
iso_forest = IsolationForest(contamination=0.005, random_state=89)
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
"""
sns.pairplot(data=df_filtrado["HighBP","HighChol","Smoker","Fruits","Education","Fruits Veggies"], hue='Diabetes_binary', diag_kind='kde')
plt.suptitle('Pairplot del dataset filtrado', y=1.02)
plt.show()
"""
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
# --- Selección del filtro de ruido según la variable de experimento ---
# Define la variable para el experimento:
USE_TOMEKLINKS = False  # Cambia a False para usar Edited Nearest Neighbors

if USE_TOMEKLINKS:
    print("Aplicando TomekLinks...")
    noise_filter = TomekLinks(sampling_strategy='auto')
else:
    print("Aplicando Edited Nearest Neighbors...")
    noise_filter = EditedNearestNeighbours(sampling_strategy='auto')

# Aplicamos el filtro de ruido sobre el dataset filtrado
X = df_filtrado.drop('Diabetes_binary', axis=1)
y = df_filtrado['Diabetes_binary']

X_res, y_res = noise_filter.fit_resample(X, y)
df_final = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=['Diabetes_binary'])], axis=1)

print("\nDataset final tras aplicar el filtro de ruido:")
print(df_final.head())
print("Dimensiones del dataset final:", df_final.shape)

# --- Visualización: Pairplot y Matriz de correlaciones ---
"""
# Pairplot usando 'Diabetes_binary' como hue
sns.pairplot(data=df_final, hue='Diabetes_binary')
plt.suptitle('Pairplot del dataset final', y=1.02)
plt.show()
"""
# Matriz de correlaciones
corr_final = df_final.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_final, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5)
plt.title('Matriz de correlaciones del dataset final')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

df_filtrado=df_final
# VAMOS A ELIMINAR LAS COLUMNAS NoDocbcCost  CholCheck AnyHealthCare
df_filtrado = df_filtrado.drop(columns=['NoDocbcCost', 'CholCheck', 'AnyHealthCare'], errors='ignore')

# Mostramos las primeras filas para confirmar que se han eliminado las columnas
print("Dataset filtrado sin las columnas NoDocbcCost, CholCheck y AnyHealthCare:")
print(df_filtrado.head())

# Seleccionamos las columnas 'Fruits' y 'Veggies'
pca_features = df_filtrado[['Fruits', 'Veggies']]

# Instanciamos PCA para extraer una única componente
pca = PCA(n_components=1, random_state=89)
df_filtrado['EatsHealthy'] = pca.fit_transform(pca_features)

# Obtenemos el rango y otras estadísticas descriptivas
min_val = df_filtrado['EatsHealthy'].min()
max_val = df_filtrado['EatsHealthy'].max()

print("Rango de valores de 'EatsHealthy' (PCA):", min_val, "a", max_val)
print("\nDescripción de 'EatsHealthy':")
print(df_filtrado['EatsHealthy'].describe())

# Seleccionamos las columnas de interés
health_features = df_filtrado[['GenHlth', 'MentHlth', 'PhysHlth']]

# Instanciamos PCA para extraer una única componente
pca_health = PCA(n_components=1, random_state=89)
df_filtrado['Health'] = pca_health.fit_transform(health_features)

# Mostramos estadísticas descriptivas de la nueva variable
print("Rango y descripción de la variable 'Health':")
print(df_filtrado['Health'].describe())

#df_filtrado = df_filtrado.drop(columns=['GenHlth', 'MentHlth', 'PhysHlth','Fruits', 'Veggies'])

# Seleccionamos las columnas numéricas del DataFrame
numeric_cols = df_filtrado.select_dtypes(include=['float64', 'int64']).columns

# Creamos una copia del DataFrame para no modificar el original
df_filtrado_scaled = df_filtrado.copy()

# Instanciamos y aplicamos el StandardScaler
scaler = StandardScaler()
df_filtrado_scaled[numeric_cols] = scaler.fit_transform(df_filtrado_scaled[numeric_cols])

# Mostramos las primeras filas del DataFrame escalado
print("Primeras filas del dataset escalado:")
print(df_filtrado_scaled.head())

# Supongamos que df_filtrado es tu dataset final ya procesado
X = df_filtrado.drop('Diabetes_binary', axis=1)
y = df_filtrado['Diabetes_binary']

# Dividimos el dataset en entrenamiento y prueba, estratificando por la variable objetivo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=89, test_size=0.2
)

# --- Bagging con conjuntos balanceados para Decision Tree ---
bbc_dt = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=89),
    n_estimators=10,                # número de modelos en el ensamble
    sampling_strategy='auto',       # balancea automáticamente según la clase minoritaria
    replacement=False,
    random_state=89
)
bbc_dt.fit(X_train, y_train)
y_pred_dt = bbc_dt.predict(X_test)

print("Resultados del modelo Decision Tree con Bagging y conjuntos balanceados:")
print(classification_report(y_test, y_pred_dt))


# --- Bagging con conjuntos balanceados para Random Forest ---
bbc_rf = BalancedBaggingClassifier(
    estimator=RandomForestClassifier(random_state=89),
    n_estimators=10,
    sampling_strategy='auto',
    replacement=False,
    random_state=89
)
bbc_rf.fit(X_train, y_train)
y_pred_rf = bbc_rf.predict(X_test)

print("Resultados del modelo Random Forest con Bagging y conjuntos balanceados:")
print(classification_report(y_test, y_pred_rf))