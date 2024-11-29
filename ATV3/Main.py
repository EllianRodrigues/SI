import pandas as pd

arquivo_csv = "dados.csv"
df = pd.read_csv(arquivo_csv, index_col=0)

normalized_df = (df - df.mean()) / df.std() # Normalizar

deviation_sum = normalized_df.abs().sum() # Calcula a soma dos desvios absolutos para cada objeto

outlier_object = deviation_sum.idxmax() # Identifica o objeto mais discrepante

print("=== Dados Normalizados ===")
print(normalized_df)
print("\n=== Soma dos Desvios Absolutos ===")
print(deviation_sum)
print(f"\nO objeto com maior probabilidade de ser defeituoso é: {outlier_object}")
print(f"\nDados do objeto defeituoso ({outlier_object}):")
print(normalized_df[outlier_object])
