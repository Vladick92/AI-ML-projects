import pandas as pd
import joblib

model=joblib.load('./wine_model.pkl')
df=pd.read_csv('../datasets/wine_val_data.csv')
df=df.drop('Unnamed: 0',axis=1)

preds=model.predict(df.drop('quality',axis=1))
preds=pd.DataFrame(preds)
preds.to_csv('../datasets/predictions.csv')