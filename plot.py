import matplotlib.pyplot as plt
from classify import classify_data

#contagem de valores gerados
classify_data(output['new_class'].value_counts().plot.bar(figsize=(10, 5),
                                            rot=0,
                                            title="# de novos valores gerados"));

                                            #distribuição do output produzido 
#conseguimos inferir a classificação novas temperaturas
#a partir de um dataset com 6 exemplos

classify_data(output.boxplot(by='new_class', figsize=(10,5)));
