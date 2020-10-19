import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression #modelo

#leitura dos dados csv

df = pd.read_csv("https://pycourse.s3.amazonaws.com/temperature.csv")
df

df.sort_values

#Extração de x e y

x, y = df[['temperatura']].values, df[['classification']].values
print("x:\n", x)
print("y:\n", y)

#pré processamento

#labelEncoder= transforma as labels [tolyo, tokyo, paris] em númerico [2,2,1]

LabelEncoder

#conversão de y para valores númerios 
le = LabelEncoder() #chamada
y = le.fit_transform(y.ravel())
print("y:\n", y)

#classificador
clf=LogisticRegression()
clf.fit(x, y)

#gerando 100 valores de temperatura
#linearmente espaçados entre 0 e 45
#predição em novos valores de temperatura

x_test = np.linspace(start=0., stop=45., num=100).reshape(-1, 1)

#predição desses valores

y_pred = clf.predict(x_test)
print(y_pred)

#conversando de y_pred para os valores originais

y_pred = le.inverse_transform(y_pred)
print(y_pred)

#output

output = {'new_temp': x_test.ravel(),
          'new_class': y_pred.ravel()}

output = pd.DataFrame(output)

#estatisticas

output.info()

#estatisticas

output.describe()

#contagem de valores gerados
output['new_class'].value_counts().plot.bar(figsize=(10, 5),
                                            rot=0,
                                            title="# de novos valores gerados");

                                            #distribuição do output produzido 
#conseguimos inferir a classificação novas temperaturas
#a partir de um dataset com 6 exemplos

output.boxplot(by='new_class', figsize=(10,5));


#sistema automático
def classify_temp():
  """Classifica o input do usuário."""

  ask = True
  while ask:
    #inpuut de temperatura
    temp = input("Insira a temperatura (graus Celsius): ")

    #transformar para numpy array
    temp = np.array(float(temp)).reshape(-1, 1)

    #realiza a classificação
    class_temp = clf.predict(temp)

    #transformação inversa para retornar a string original
    class_temp = le.inverse_transform(class_temp)

    #classificação
    print(f"A classificação da temperatura {temp.ravel()[0]} é:", class_temp[0])

    #perguntar
    ask = input("Nova classificação (y/n): ") == 'y'

#rodando o programa
classify_temp()

#validar o modelo

