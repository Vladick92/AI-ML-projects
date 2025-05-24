import pandas as pd
import joblib

model=joblib.load('./car_model.pkl')
df=pd.read_csv('../datasets/cars_val_data.csv')
df=df.drop('Unnamed: 0',axis=1)

preds=model.predict(df.drop('price',axis=1))
preds=pd.DataFrame(preds)
preds.to_csv('../datasets/predictions.csv')