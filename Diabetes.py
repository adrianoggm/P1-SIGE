import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Habilitamos IterativeImputer (característica experimental)
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Importamos los filtros de ruido de imbalanced-learn
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours,CondensedNearestNeighbour
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

plt.figure(figsize=(10,6))
ax = sns.violinplot(x='Diabetes_binary', y='BMI', data=df, palette='Set2', inner='quartile')
plt.title('Distribución de BMI respecto a Diabetes')
plt.xlabel('Diabetes (0 = No, 1 = Sí)')
plt.ylabel('BMI')

# Calculamos el número de muestras en cada clase
counts = df['Diabetes_binary'].value_counts().sort_index()
# Obtenemos el valor máximo de BMI para posicionar las anotaciones
y_max = df['BMI'].max()

# Agregamos las anotaciones en el gráfico
for i, count in enumerate(counts):
    ax.text(i, y_max, f'N={count}', horizontalalignment='center',
            fontsize=12, color='black', weight='semibold')

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

# Redondeamos las columnas que deben ser enteros
binary_columns = ['CholCheck', 'HeartDiseaseorAttack', 'NoDocbcCost','Income']
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
# Definir el método de filtrado de ruido a utilizar: opciones 'TomekLinks', 'ENN' o 'CNN'
noise_filter_method = 'ENN'  # Cambia a 'TomekLinks' o 'ENN' según se desee

if noise_filter_method == 'TomekLinks':
    print("Aplicando TomekLinks...")
    noise_filter = TomekLinks(sampling_strategy='auto')
elif noise_filter_method == 'ENN':
    print("Aplicando Edited Nearest Neighbours...")
    noise_filter = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=4)
elif noise_filter_method == 'CNN':
    print("Aplicando Condensed Nearest Neighbours (CNN)...")
    noise_filter = CondensedNearestNeighbour(sampling_strategy='auto') # hasta el infinito
else:
    raise ValueError("Método de filtrado no reconocido. Elija 'TomekLinks', 'ENN' o 'CNN'.")

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

# Ejemplo de uso: Ajustar y transformar un conjunto de datos (X, y)
# X_filtered, y_filtered = noise_filter.fit_resample(X, y)
# print("Tamaño original:", len(y))
# print("Tamaño filtrado:", len(y_filtered))

# =============================================================================
# Definimos los límites de los bins y las etiquetas deseadas:
bins = [0, 4, 6, 8]  # Los límites: 1-4, 5-6 y 7-8
labels = [0, 1, 2]   # 0: clase baja, 1: clase media, 2: clase alta

# Creamos la nueva columna 'Income_class' usando pd.cut()
df_filtrado['Income_class'] = pd.cut(df_filtrado['Income'], bins=bins, labels=labels, include_lowest=True)
bins = [-np.inf, 18.5, 25, 30, 42.1,np.inf]
labels = [0, 1, 2, 3, 4]

# Creamos la nueva columna 'BMI_category' en el DataFrame:
df_filtrado['BMI_category'] = pd.cut(df_filtrado['BMI'], bins=bins, labels=labels, right=False)

# Mostramos las primeras filas para verificar la transformación
print(df_filtrado[['Income', 'Income_class']].head())


# VAMOS A ELIMINAR LAS COLUMNAS NoDocbcCost  CholCheck AnyHealthcare ,'HvyAlcoholConsump'
df_filtrado = df_filtrado.drop(columns=['NoDocbcCost',  'AnyHealthcare','HvyAlcoholConsump','Smoker'], errors='ignore')

# Mostramos las primeras filas para confirmar que se han eliminado las columnas
print("Dataset filtrado sin las columnas NoDocbcCost, CholCheck y AnyHealthCare:")
print(df_filtrado.head())
# Creamos una instancia de StandardScaler para normalizar las variables
scaler = StandardScaler()

# =============================================================================
# Extracción de la variable 'EatsHealthy'
# =============================================================================
# Seleccionamos y normalizamos las columnas 'Fruits' y 'Veggies'
pca_features = df_filtrado[['Fruits', 'Veggies']]
pca_features_scaled = scaler.fit_transform(pca_features)

# Aplicamos PCA para extraer una única componente
pca = PCA(n_components=1, random_state=89)
df_filtrado['EatsHealthy'] = pca.fit_transform(pca_features_scaled)

# Mostramos el rango y estadísticas descriptivas de 'EatsHealthy'
min_val = df_filtrado['EatsHealthy'].min()
max_val = df_filtrado['EatsHealthy'].max()
print("Rango de valores de 'EatsHealthy' (PCA):", min_val, "a", max_val)
print("\nDescripción de 'EatsHealthy':")
print(df_filtrado['EatsHealthy'].describe())

# =============================================================================
# Extracción de la variable 'Health'
# =============================================================================
# Seleccionamos y normalizamos las columnas 'GenHlth', 'MentHlth' y 'PhysHlth'
health_features = df_filtrado[['GenHlth', 'MentHlth', 'PhysHlth',]]
health_features_scaled = scaler.fit_transform(health_features)

# Aplicamos PCA para extraer una única componente
pca_health = PCA(n_components=1, random_state=89)
df_filtrado['Health'] = pca_health.fit_transform(health_features_scaled)

# Mostramos estadísticas descriptivas de 'Health'
print("Rango y descripción de la variable 'Health':")
print(df_filtrado['Health'].describe())

# =============================================================================
# Visualización de la matriz de correlaciones (antes de agregar SocioEconomics)
# =============================================================================
corr_filtrado = df_filtrado.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_filtrado, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title('Matriz de correlaciones del dataset filtrado')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =============================================================================
# Extracción de la variable 'SocioEconomics'
# =============================================================================
# Seleccionamos y normalizamos las columnas 'Income', 'Age' y 'Education'
socioeconomic_features = df_filtrado[['Income', 'Age', 'Education']]
socioeconomic_features_scaled = scaler.fit_transform(socioeconomic_features)

# Aplicamos PCA para extraer una única componente
pca_socioeconomic = PCA(n_components=1, random_state=89)
df_filtrado['SocioEconomics'] = pca_socioeconomic.fit_transform(socioeconomic_features_scaled)

# Mostramos estadísticas descriptivas de 'SocioEconomics'
print("Rango y descripción de la variable 'SocioEconomics':")
print(df_filtrado['SocioEconomics'].describe())

# =============================================================================
# Visualización de la matriz de correlaciones (después de agregar SocioEconomics)
# =============================================================================
corr_filtrado = df_filtrado.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_filtrado, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title('Matriz de correlaciones del dataset filtrado')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# =============================================================================
# Extracción de la variable 'HeartRiskFactor'
# =============================================================================
# Seleccionamos y normalizamos las columnas 'Stroke', 'HeartDiseaseorAttack' y 'PhysActivity'
heart_features = df_filtrado[['Stroke', 'HeartDiseaseorAttack','CholCheck','PhysActivity','BMI']]
heart_features_scaled = scaler.fit_transform(heart_features)

# Aplicamos PCA para extraer una única componente
pca_heart = PCA(n_components=1, random_state=89)
df_filtrado['HeartRiskFactor'] = pca_heart.fit_transform(heart_features_scaled)

# Mostramos estadísticas descriptivas de 'HeartRiskFactor'
print("Rango y descripción de la variable 'HeartRiskFactor':")
print(df_filtrado['HeartRiskFactor'].describe())

# =============================================================================
# Visualización de la matriz de correlaciones (después de agregar HeartRiskFactor)
# =============================================================================
corr_filtrado = df_filtrado.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_filtrado, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title('Matriz de correlaciones del dataset filtrado')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# Calcula la correlación de cada variable con Diabetes_binary
# Separamos las características y la variable objetivo
X = df_filtrado.drop('Diabetes_binary', axis=1)
y = df_filtrado['Diabetes_binary']

# Entrenamos un clasificador Random Forest
rf = RandomForestClassifier(random_state=89)
rf.fit(X, y)

# Extraemos las importancias de cada característica
importances = rf.feature_importances_

# Creamos un DataFrame para ordenarlas
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Importancia de las características:")
print(importance_df)

# Extraemos el vector de características ordenado según la importancia
feature_vector = importance_df['feature'].tolist()
print("\nVector de características ordenado por importancia:")
print(feature_vector)



# Seleccionamos una muestra de 10,000 sujetos para el gráfico
sample = df_filtrado.sample(100, random_state=89)

# Creamos una nueva columna de colores basada en la variable "Diabetes_binary"
sample['color'] = sample['Diabetes_binary'].map({0: 'blue', 1: 'red'})

marker_dict = {
    1: 'o',  # círculo
    2: 's',  # cuadrado
    3: 'D',  # diamante
    4: 'v',  # triángulo abajo
    5: '^'   # triángulo arriba
}

plt.figure(figsize=(10,6))

# Iteramos por cada clase de GenHlth y graficamos el subconjunto correspondiente
for genhlth, marker in marker_dict.items():
    subset = sample[sample['GenHlth'] == genhlth]
    plt.scatter(subset['SocioEconomics'], subset['BMI'], 
                c=subset['color'], alpha=0.5, marker=marker, 
                label=f'GenHlth: {genhlth}')

plt.xlabel('SocioEconomics')
plt.ylabel('BMI')
plt.title('Relación entre BMI y SocioEconomics según Diabetes y GenHlth')
plt.legend(title='GenHlth', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

df_filtrado = df_filtrado.drop(columns=['Fruits','Veggies','Education'], errors='ignore')
# =============================================================================
# Normalización final de todas las columnas numéricas del DataFrame
# =============================================================================
numeric_cols = df_filtrado.select_dtypes(include=['float64', 'int64']).columns
df_filtrado_scaled = df_filtrado.copy()
df_filtrado_scaled[numeric_cols] = scaler.fit_transform(df_filtrado_scaled[numeric_cols])

# Mostramos las primeras filas del DataFrame escalado
print("Primeras filas del dataset escalado:")
print(df_filtrado_scaled.head())

# Supongamos que df_filtrado es tu dataset final ya procesado
X = df_filtrado.drop('Diabetes_binary', axis=1)
y = df_filtrado['Diabetes_binary']

# Dividimos el dataset en entrenamiento y prueba, estratificando por la variable objetivo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=89, test_size=0.20
)
# Definimos la variable de experimento para seleccionar el filtro de ruido
"""
USE_TOMEKLINKS = False  # Cambia a True para usar TomekLinks

if USE_TOMEKLINKS:
    print("Aplicando TomekLinks sobre el conjunto train...")
    noise_filter = TomekLinks(sampling_strategy='auto')
else:
    print("Aplicando Edited Nearest Neighbours sobre el conjunto train...")
    noise_filter = EditedNearestNeighbours(sampling_strategy='auto')

# Aplicamos el filtro de ruido únicamente sobre el conjunto de entrenamiento
X_train, y_train = noise_filter.fit_resample(X_train, y_train)
print("Dimensiones del conjunto train despues:", X_train.shape)"
"""

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

# --- Balanced Bagging con Random Forest ---
bbc_rf = BalancedBaggingClassifier(
    estimator=RandomForestClassifier(
        n_estimators=200,
        criterion='entropy',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=89
    ),
    n_estimators=10,
    sampling_strategy='auto',
    replacement=False,
    random_state=89
)
bbc_rf.fit(X_train, y_train)
y_pred_rf = bbc_rf.predict(X_test)
print("Resultados del modelo Random Forest con Balanced Bagging:")
print(classification_report(y_test, y_pred_rf))


# --- Balanced Bagging con AdaBoostClassifier ---
bbc_ada = BalancedBaggingClassifier(
    estimator=AdaBoostClassifier(n_estimators=100, random_state=89),
    n_estimators=10,
    sampling_strategy='auto',
    replacement=False,
    random_state=89
)
bbc_ada.fit(X_train, y_train)
y_pred_ada = bbc_ada.predict(X_test)
print("Resultados del modelo AdaBoost con Balanced Bagging:")
print(classification_report(y_test, y_pred_ada))


# --- Balanced Bagging con XGBoost ---
bbc_xgb = BalancedBaggingClassifier(
    estimator=xgb.XGBClassifier(n_estimators=100, random_state=89, eval_metric='logloss'),
    n_estimators=10,
    sampling_strategy='auto',
    replacement=False,
    random_state=89
)
bbc_xgb.fit(X_train, y_train)
y_pred_xgb = bbc_xgb.predict(X_test)
print("Resultados del modelo XGBoost con Balanced Bagging:")
print(classification_report(y_test, y_pred_xgb))


# --- Balanced Bagging con Regresión Logística ---
bbc_lr = BalancedBaggingClassifier(
    estimator=LogisticRegression(random_state=89, max_iter=1000),
    n_estimators=10,
    sampling_strategy='auto',
    replacement=False,
    random_state=89
)
bbc_lr.fit(X_train, y_train)
y_pred_lr = bbc_lr.predict(X_test)
print("Resultados del modelo Logistic Regression con Balanced Bagging:")
print(classification_report(y_test, y_pred_lr))


# --- Cálculo de matrices de confusión ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_ada = confusion_matrix(y_test, y_pred_ada)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_lr = confusion_matrix(y_test, y_pred_lr)

# --- Visualización: Matrices de confusión para los 4 modelos (excluyendo el modelo basado en Decision Tree) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Matriz de confusión para Random Forest
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0,0])
axes[0,0].set_title("Random Forest con Balanced Bagging")
axes[0,0].set_xlabel("Predicción")
axes[0,0].set_ylabel("Valor Real")

# Matriz de confusión para AdaBoost
sns.heatmap(cm_ada, annot=True, fmt="d", cmap="Blues", ax=axes[0,1])
axes[0,1].set_title("AdaBoost con Balanced Bagging")
axes[0,1].set_xlabel("Predicción")
axes[0,1].set_ylabel("Valor Real")

# Matriz de confusión para XGBoost
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", ax=axes[1,0])
axes[1,0].set_title("XGBoost con Balanced Bagging")
axes[1,0].set_xlabel("Predicción")
axes[1,0].set_ylabel("Valor Real")

# Matriz de confusión para Logistic Regression
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[1,1])
axes[1,1].set_title("Logistic Regression con Balanced Bagging")
axes[1,1].set_xlabel("Predicción")
axes[1,1].set_ylabel("Valor Real")

plt.tight_layout()
plt.show()