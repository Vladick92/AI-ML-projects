import pandas as pd
import joblib

clas_model=joblib.load('./label_model.pkl')
reg_model=joblib.load('./bmi_model.pkl')

df=pd.read_csv('../datasets/obesity_val_data.csv')
df=df.drop('Unnamed: 0',axis=1)

pred_bmis=reg_model.predict(df.drop(['Label','BMI'],axis=1))
pred_labels=clas_model.predict(df.drop(['Label','BMI'],axis=1))

result=pd.DataFrame({
    "Predicted_BMI": pred_bmis,
    "Predicted_Label": pred_labels
})
result.to_csv('../datasets/predictions.csv')