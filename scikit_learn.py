import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
from sklearn.preprocessing import LabelEncoder

LabelEncoder