import pandas as pd
import joblib

model=joblib.load('./bank_model.pkl')
df=pd.read_csv('../datasets/loan_val_data.csv')
df=df.drop('Unnamed: 0',axis=1)
preds=model.predict(df.drop('Credit_Worthiness',axis=1))
preds=pd.DataFrame(preds)
preds.to_csv('../datasets/predictions.csv')