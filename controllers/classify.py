from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from pre_processing import processing_data

def classify_data():
  x, y = processing_data()
  le = LabelEncoder()
  # print("aquii",x, y)
  clf=LogisticRegression()  
  clf.fit(x, y)
  # print("aquii",x, y)

  # #gerando 100 valores de temperatura
  # #linearmente espaçados entre 0 e 45
  # #predição em novos valores de temperatura

  x_test = np.linspace(start=0., stop=45., num=100).reshape(-1, 1)

  # #predição desses valores

  y_pred = clf.predict(x_test)
  print("aqui",y_pred)

  # #conversando de y_pred para os valores originais

  y_pred = le.inverse_transform(y_pred)
  print("e aqui",y_pred)

  # # #output

  # output = {'new_temp': x_test.ravel(),
  #           'new_class': y_pred.ravel()}

  # output = pd.DataFrame(output)

  # # #estatisticas

  # output.info()

  # # #estatisticas

  # output.describe()

  return x, y


# classify_data()