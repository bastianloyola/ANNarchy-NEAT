import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos Breast Cancer
data = load_breast_cancer()
X, y = data.data, data.target

# Dividir los datos, seleccionando un 30% estratificado para optimización
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)

# Crear un DataFrame con los datos seleccionados
df_sample = pd.DataFrame(X_sample, columns=data.feature_names)
df_sample['target'] = y_sample

# Guardar el subconjunto de datos en un archivo CSV
df_sample.to_csv('breast_cancer_sample.csv', index=False)
print("Datos guardados en 'breast_cancer_sample.csv'")
print(len(df_sample))
