import pandas as pd 
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('../datasets/obesity_train_data.csv')
df=df.drop('Unnamed: 0',axis=1)

reg_model=GradientBoostingRegressor(
    alpha= 0.9,
    ccp_alpha= 0.0,
    criterion= 'friedman_mse',
    init= None,
    learning_rate= 0.2,
    loss= 'squared_error',
    max_depth= 5,
    max_features= None,
    max_leaf_nodes= None,
    min_impurity_decrease= 0.0,
    min_samples_leaf= 1,
    min_samples_split= 2,
    min_weight_fraction_leaf= 0.0,
    n_estimators= 200,
    n_iter_no_change= None,
    random_state= 42,
    subsample= 0.6,
    tol= 0.0001,
    validation_fraction= 0.1,
    verbose= 0,
    warm_start= False
    )
reg_model.fit(df.drop(['Label','BMI'],axis=1),df['BMI'])

clas_model=KNeighborsClassifier(
    algorithm= 'auto',
    leaf_size= 30,
    metric= 'manhattan',
    metric_params= None,
    n_neighbors= 7,
    p= 1,
    weights= 'uniform'
)
clas_model.fit(df.drop(['Label','BMI'],axis=1),df['Label'])
joblib.dump(clas_model,'./label_model.pkl')
joblib.dump(reg_model,'./bmi_model.pkl')