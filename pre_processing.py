from sklearn.preprocessing import LabelEncoder
from resource.data_frame import read_df

def processing_data():
    df = read_df()
    x, y = df[['temperatura']].values, df[['classification']].values

    le = LabelEncoder() #chamada
    y = le.fit_transform(y.ravel())
    # print("y:\n", y)

    return x,y