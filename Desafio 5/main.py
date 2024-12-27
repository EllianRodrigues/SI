import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def intervalo_para_media(x):
    if isinstance(x, str):
        limites = x.strip('()[]').split(', ')
        return (float(limites[0]) + float(limites[1])) / 2  # Média entre os limites do intervalo
    return x  # Caso não seja intervalo, retorna o valor original


# Abrir o CSV
df = pd.read_csv('amostra_praso - amostra_praso.csv')
# Colunas importantes sem 'serasa_credores'
# Adicionar 'fonte_cliente' às colunas importantes
colunas_importantes = [
    'capital_social',
    'idade_cnpj',
    'serasa_contagem_negativacoes',
    'serasa_contagem_protestos',
    'serasa_idade_divida_mais_recente',
    'serasa_socio_tem_negativacao',
    'inadimplente',
    'fonte_cliente'
]

# Manter apenas as colunas importantes
df = df[colunas_importantes]

# Aplicar One-Hot Encoding na coluna 'fonte_cliente'
df = pd.get_dummies(df, columns=['fonte_cliente'], prefix='fonte_cliente')

# Tratar valores ausentes
df['serasa_idade_divida_mais_recente'] = df['serasa_idade_divida_mais_recente'].fillna(0)  # Sem dívidas

# Aplicar a função para as colunas que possuem intervalos
df['capital_social'] = df['capital_social'].apply(intervalo_para_media)
df['idade_cnpj'] = df['idade_cnpj'].apply(intervalo_para_media)

# Verificar os dados
print("Amostra dos dados processados:")
print(df.head())
print(df.shape)
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Separando variáveis independentes (X) e a variável alvo (y)
X = df.drop(columns=['inadimplente'])
y = df['inadimplente']
# Dividindo os dados em 80% treino e 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
# Ajustando e transformando os dados de treinamento
X_train_scaled = scaler.fit_transform(X_train)
# Transformando os dados de teste com os parâmetros do treinamento
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)
# Acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.4f}')
# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print(f'Matriz de Confusão:\n{cm}')
# Relatório de Classificação
report = classification_report(y_test, y_pred)
print(f'Relatório de Classificação:\n{report}')



#######################################################################
# # Abrir o novo CSV
# df_novo = pd.read_csv('teste.csv')

# # Manter apenas as colunas importantes
# colunas_importantes = [
#     'capital_social',
#     'idade_cnpj',
#     'serasa_contagem_negativacoes',
#     'serasa_contagem_protestos',
#     'serasa_idade_divida_mais_recente',
#     'serasa_socio_tem_negativacao',
#     'fonte_cliente'
# ]
# df_novo = df_novo[colunas_importantes]

# # Aplicar One-Hot Encoding na coluna 'fonte_cliente'
# df_novo = pd.get_dummies(df_novo, columns=['fonte_cliente'], prefix='fonte_cliente')

# # Forçar consistência nos nomes das colunas
# df_novo.columns = [col.replace('fonte_cliente_fonte', 'fonte_cliente_Fonte') for col in df_novo.columns]

# # Tratar valores ausentes
# df_novo['serasa_idade_divida_mais_recente'] = df_novo['serasa_idade_divida_mais_recente'].fillna(0)  # Sem dívidas

# # Aplicar a função para as colunas que possuem intervalos
# df_novo['capital_social'] = df_novo['capital_social'].apply(intervalo_para_media)
# df_novo['idade_cnpj'] = df_novo['idade_cnpj'].apply(intervalo_para_media)

# # Ajustar e transformar os dados de df_novo
# df_novo_scaled = scaler.transform(df_novo)

# # Realizar a previsão com as características corretas
# new_predictions = model.predict(df_novo_scaled)

# # Exibir as previsões
# print(new_predictions)

# df_novo = pd.read_csv('teste.csv')
# capital_social_lista = df_novo['inadimplente'].tolist()
# capital_social_string = ' '.join(map(str, capital_social_lista))
# print(capital_social_string)




