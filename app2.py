from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from controllers.classify import classify_data
import numpy as np

#sistema automático
def classify_temp():
  """Classifica o input do usuário."""
  x, y = classify_data()
  print("x e y", x, y)
  clf = LogisticRegression()
  le = LabelEncoder()
  clf.fit(x, y)
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

classify_temp()