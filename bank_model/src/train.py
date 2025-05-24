import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

model=GradientBoostingClassifier(
    random_state=42,
    ccp_alpha= 0.0, 
    criterion= 'friedman_mse', 
    init= None, 
    learning_rate= 0.05, 
    loss= 'log_loss', 
    max_depth= 10, 
    max_features= 'sqrt', 
    max_leaf_nodes= None, 
    min_impurity_decrease= 0.0, 
    min_samples_leaf= 1, 
    min_samples_split= 10, 
    min_weight_fraction_leaf= 0.0, 
    n_estimators= 100, 
    n_iter_no_change= None, 
    subsample= 0.8, 
    tol= 0.0001, 
    validation_fraction= 0.1, 
    verbose= 0, 
    warm_start= False
)

df=pd.read_csv('../datasets/loan_train_data.csv')
df=df.drop('Unnamed: 0',axis=1)
model.fit(df.drop('Credit_Worthiness',axis=1),df['Credit_Worthiness'])
joblib.dump(model,'./bank_model.pkl')