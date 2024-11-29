import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

arquivo_csv = "dados.csv"
df = pd.read_csv(arquivo_csv, index_col=0)  # primeira coluna é o índice

print("Dados carregados:")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)

# normalizar
scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(df)

#verificar as anomalias nas linhas com trashold 95 
media_band = dados_normalizados.mean(axis=1)
threshold_band = np.percentile(media_band, 95)
anomalia_band = df.index[np.where(media_band > threshold_band)]

print(f"\nBands com anomalias: {anomalia_band}")

#verificar as anomalias nas colunas com trashold 95 
media_object = dados_normalizados.mean(axis=0)
threshold_object = np.percentile(media_object, 95)
anomalia_object = df.columns[np.where(media_object > threshold_object)]

print(f"\nObjetos anômalos: {anomalia_object}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(media_band, marker="o", label="Média por Banda")
plt.axhline(threshold_band, color="r", linestyle="--", label="Limiar de Anomalia")
plt.title("Análise por Bandas")
plt.xlabel("Bands")
plt.ylabel("Média Normalizada")
plt.legend()


plt.subplot(1, 2, 2)
plt.bar(range(len(media_object)), media_object, label="Média por Objeto")
plt.axhline(threshold_object, color="r", linestyle="--", label="Limiar de Anomalia")
plt.title("Análise por Objetos")
plt.xlabel("Objetos")
plt.ylabel("Média Normalizada")
plt.legend()
plt.tight_layout()
plt.show()