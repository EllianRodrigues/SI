import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def intervalo_para_media(x):
    if isinstance(x, str):
        limites = x.strip('()[]').split(', ')
        return (float(limites[0]) + float(limites[1])) / 2
    return x

# Abrir o CSV
df = pd.read_csv('amostra_praso - amostra_praso.csv')
# Colunas importantes sem 'serasa_credores'
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
df = df[colunas_importantes]

# Aplicar One-Hot Encoding na coluna 'fonte_cliente'
df = pd.get_dummies(df, columns=['fonte_cliente'], prefix='fonte_cliente')

# Tratar valores ausentes
df['serasa_idade_divida_mais_recente'] = df['serasa_idade_divida_mais_recente'].fillna(0)

# Aplicar a função para as colunas que possuem intervalos
df['capital_social'] = df['capital_social'].apply(intervalo_para_media)
df['idade_cnpj'] = df['idade_cnpj'].apply(intervalo_para_media)

# Separando variáveis independentes (X) e a variável alvo (y)
X = df.drop(columns=['inadimplente'])
y = df['inadimplente']

# Dividindo os dados em 80% treino e 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definindo a rede neural
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Avaliando o modelo
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Acurácia: {accuracy:.4f}')
# Fazendo previsões
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Matriz de Confusão e Relatório de Classificação
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Matriz de Confusão:\n{cm}')
print(f'Relatório de Classificação:\n{report}')
