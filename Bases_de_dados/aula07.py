import pandas as pd
import numpy as np

dados = pd.read_json(path_or_buf='imoveis.json',orient='columns') #leitura do arquivo json

# print(dados)
dados.ident[0]
# print(dados.ident[0])
dados.listing[0]
# print(dados.listing[0])

dados_lista1 = pd.json_normalize(dados.ident) # tranforma json em df

dados_lista1.head()

dados_lista2 = pd.json_normalize(dados.listing,sep='_')

dados_lista2.head()

dados_imoveis = pd.concat([dados_lista1,dados_lista2],axis=1)

dados_imoveis.head()

dados_imoveis.shape # retorna as dimensoes do dataframe

for coluna in dados_imoveis.columns:
    print('----'*10)
    print(dados_imoveis[coluna].value_counts())
    
    #Pré processamento dos dados
#Aplicando filtros no dataset
filtro = (dados_imoveis['types_usage'] == 'Residencial')&(dados_imoveis['address_city'] == 'Rio de Janeiro')

dados_imoveis= dados_imoveis[filtro]

dados_imoveis.head()

dados_imoveis.info() # verifica as informações dos dados

# reset do index 
dados_imoveis.reset_index(drop=True, inplace=True)

dados_imoveis.info(verbose= False)

#transformação dos tipos de dados
dados_imoveis = dados_imoveis.astype({
    'prices_price':'float64',
    'prices_tax_iptu':'float64',
    'prices_tax_condo':'float64',
    'features_usableAreas':'int64',
    'features_totalAreas':'int64'
})

dados_imoveis.info()

#Lidando com dados nulos
dados_imoveis['address_zone'].value_counts()

dados_imoveis['address_zone'] = dados_imoveis['address_zone'].replace('',np.nan) #substitui o espaço vazio por nan

dados_imoveis['address_zone'].value_counts()

dados_imoveis['address_zone'].isnull().value_counts()

dados_imoveis.info()

dados_imoveis.head()

dici = dados_imoveis[~dados_imoveis['address_zone'].isna()].drop_duplicates(subset=['address_neighborhood']).to_dict('records') # remove valores duplicados
print(dici)

dados_imoveis['address_zone'].isnull().sum()

dic_zonas = {dic['address_neighborhood']:dic['address_zone']for dic in dici} #criando dicionario para associar

print(dic_zonas)

#Associando o bairro com a zona
for bairro, zona in dic_zonas.items():
    dados_imoveis.loc[dados_imoveis['address_neighborhood']==bairro,'address_zone']=zona
    
dados_imoveis.head()

dados_imoveis['address_zone'].isnull().sum()

dados_imoveis['prices_tax_condo'].isnull().sum()

dados_imoveis['prices_tax_iptu'].isnull().sum()

dados_imoveis['prices_tax_condo'].fillna(0,inplace = True)
dados_imoveis['prices_tax_iptu'].fillna(0,inplace = True)

print(f"Total de valores nulos tax condo:{dados_imoveis['prices_tax_condo'].isnull().sum()}")
print(f"Total de valores nulos tax iptu:{dados_imoveis['prices_tax_iptu'].isnull().sum()}")

#Remove colunas nao uteis para criar o modelo
dados_imoveis.drop(['customerID','source','types_usage','address_city',
                    'address_location_lon','address_location_lat','address_neighborhood'],axis=1, inplace= True)

#criando dicionario para renomear as colunas
dic_colunas={
    'types_unit':'unit','address_zone': 'zone','prices_price': 'price',
    'prices_tax_condo':'tax_condo','prices_tax_iptu':'tax_iptu','features_bedrooms':'bedrooms',
    'features_bathrooms':'bathrooms','features_suites':'suites','features_parkingSpaces':'parkingSpaces',
    'features_usableAreas':'usableAreas', 'features_totalAreas':'totalAreas', 'features_floors':'floors',
    'features_unitsOnTheFloor': 'unitsOnTheFloor','features_unitFloor':'unitFloor'
}

#renomeando as colunas
dados_imoveis= dados_imoveis.rename(dic_colunas, axis =1) #renomeia as colunas conforme o dicionario criado
dados_imoveis.head()
col_n = dados_imoveis.select_dtypes(include=['number'])

correlacao = col_n.corr()

print(correlacao)

import matplotlib.pyplot as plt
import seaborn as sns

#Visualização das correlações
cores = sns.color_palette('light:salmon', as_cmap = True) # personalização das cores
mask = np.zeros_like(correlacao)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    f,ax= plt.subplots(figsize=(13,8))
    ax = sns.heatmap(correlacao, cmap=cores, mask = mask, square = True, fmt = '.2f',annot = True)


sns.heatmap(correlacao, cmap='crest')


plt.figure(figsize=(13,8))
mask = np.zeros_like(correlacao)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlacao, cmap='crest', mask = mask, square = True, fmt = '.2f',annot = True)



ax = sns.histplot(data = dados_imoveis, x='price', kde= True)
ax.figure.set_size_inches(20,10)
ax.set_title('Histograma de preços')
ax.set_xlabel('Preços')
ax.set_xlabel('Preço')
plt.show()

from sklearn.preprocessing import FunctionTransformer #importar metodo para realizar a transformação dos dados
transformer = FunctionTransformer(np.log1p, validate=True)

dados_transformados = transformer.transform(dados_imoveis.select_dtypes(exclude=['object'])) #exclui dados que não sao numéricos

colunas_dados_tranformados = dados_imoveis.select_dtypes(exclude=['object']).columns

df_transformado = pd.concat([dados_imoveis.select_dtypes(include=['object']), pd.DataFrame(dados_transformados, columns=colunas_dados_tranformados)], axis=1)
df_transformado.head()


#depois da transformação
col_n = dados_imoveis.select_dtypes(include=['number']) # selecionando apenas colunas com numeros
correlacao_transformado = col_n.corr()
mask = np.zeros_like(correlacao_transformado)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(13, 8))
    ax = sns.heatmap(correlacao_transformado, cmap=cores, mask=mask, square=True, fmt='.2f', annot=True)
    ax.set_title('Correlação entre variáveis - Tranformação Log', fontsize=20)

plt.show()
#depois (distrib. simétrica)
ax = sns.histplot(data=df_transformado, x='price', kde=True)
ax.figure.set_size_inches(20, 10)
ax.set_title('Histograma de preços')
ax.set_xlabel('Preço')

plt.show()
variaveis_categoricas = df_transformado.select_dtypes(include=['object']).columns #variaveis categóricas

df_dummies = pd.get_dummies(df_transformado[variaveis_categoricas]) # cria uma nova coluna com variaveis categóricas
df_dummies.head()

dados_imoveis_dummies = pd.concat([df_transformado.drop(variaveis_categoricas, axis=1), df_dummies], axis=1) 
dados_imoveis_dummies.head()

dados_imoveis.head()

#Ajuste e previsao
#variáveis explanatórias (independentes)
X = dados_imoveis_dummies.drop('price', axis=1)

#variável dependente / Variavel dependente
y = dados_imoveis_dummies['price']

from sklearn.model_selection import train_test_split #Dividir o conjunto de dados para treino e teste

#divisão em conjunto de treino e teste
#random_state - estado de aleatoriedade
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42) #função para dividir o conjunto de dados

from sklearn.linear_model import LinearRegression # importa o modelo de regressao linear

#Instanciando o modelo
lr = LinearRegression()
df = X_teste[0:6]
#treino
lr.fit(X_treino, y_treino) #treino do modelo

#teste
previsao_lr = lr.predict(X_teste)
print(X_teste)

np.expm1(7.49)
dados_imoveis.head(5)
print(previsao_lr)

np.expm1(12.45)
np.expm1(13.13161073)

np.expm1(13.25768024)
#importar a biblioteca para calcular a métrica r2_score
from sklearn.metrics import r2_score 

r2_lr = r2_score(y_teste, previsao_lr)
r2_lr

from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
r2_score(y_true, y_pred)


from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)

from sklearn.metrics import mean_squared_error

y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]

mean_squared_error(y_true, y_pred, squared=True)


from sklearn.metrics import mean_squared_error

y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]

mean_squared_error(y_true, y_pred, squared=False)

from sklearn.metrics import mean_absolute_percentage_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mean_absolute_percentage_error(y_true, y_pred)


from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_teste, previsao_lr)
mape


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_teste, previsao_lr)
mse

from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)


from sklearn.metrics import mean_absolute_percentage_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_percentage_error(y_true, y_pred)


#arvore de decisão para regressao
from sklearn.tree import DecisionTreeRegressor


#Instanciando o modelo
dtr = DecisionTreeRegressor(random_state=42,max_depth=5) #random_state garante que a estrutura da árvore será reprodutivel
#max_depth define o tamanho da arvore

#treino
dtr.fit(X_treino,y_treino)


#teste
previsao_dtr = dtr.predict(X_teste)


previsao_dtr


np.expm1(13.55136531)


from yellowbrick.regressor import PredictionError
fig,ax = plt.subplots(figsize=(10,10))
pev = PredictionError(dtr)
pev.fit(X_treino,y_treino)
pev.score(X_teste,y_teste)
pev.poof()

#Metrica
r2_dtr = r2_score(y_teste,previsao_dtr)
r2_dtr

#Metódo Ensemble
#Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, max_depth=5, n_estimators=10)

rf.fit(X_treino,y_treino)

previsao_rf = rf.predict(X_teste)


previsao_rf

np.expm1(13.53707348)


from yellowbrick.regressor import PredictionError

from yellowbrick.regressor import PredictionError
fig, ax = plt.subplots(figsize=(10,10))
pev = PredictionError(rf)
pev.fit(X_treino,y_treino)
pev.score(X_teste, y_teste)
pev.poof()

#Metricas de desempenho
r2_rf = r2_score(y_teste,previsao_rf)
r2_rf


metricas_modelo_ML = pd.DataFrame({
    'Modelo': ['Regressao Linear', 'Árvore de decisão','Random Forest'],
     'Metricas': ['R2','MSE','MAE']
})

from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error


def obter_metrica(modelo, X_teste, y_teste, nome):
    predict = modelo.predict(X_teste)
    df_metricas = pd.DataFrame({
        'R2':[r2_score(y_teste,predict)],
        'MSE':[mean_squared_error(y_teste,predict)],
        'MAE':[mean_absolute_error(y_teste,predict)]
    },index=[nome])
    return df_metricas


def tabela_metricas(modelo_reg_linear, modelo_dt, modelo_rf, X_teste, y_teste):
    df_metricas_reg_linear = obter_metrica(modelo_reg_linear, X_teste, y_teste, 'Linear Regression')
    df_metricas_dt = obter_metrica(modelo_dt, X_teste, y_teste, 'Decision Tree Regression')
    df_metricas_rf = obter_metrica(modelo_rf, X_teste, y_teste, 'Random Forest Regression')

    return pd.concat([df_metricas_reg_linear, df_metricas_dt, df_metricas_rf])


tabela_metricas(lr, dtr, rf, X_teste, y_teste)


metricas_modelo_ML.head()


dados_imoveis_dummies.head(10)

X_teste[0:4]

df = X_teste[0:6]

lr.predict(df)[2]


dtr.predict(df)[2]


X_teste.head()

valor_real = np.expm1(lr.predict(df)[2])
valor_real

valor_real = np.expm1(dtr.predict(df)[2])
valor_real

dados_imoveis.head(6)

dados_imoveis_dummies.head(10)

np.expm1(10.82	)