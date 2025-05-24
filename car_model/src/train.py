import joblib
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

df=pd.read_csv('../datasets/cars_train_data.csv')
df=df.drop('Unnamed: 0',axis=1)

model=DecisionTreeRegressor(
    random_state=42,
    max_depth= 14,
    max_features= 'sqrt',
    max_leaf_nodes= 50,
    min_samples_leaf= 1,
    min_samples_split= 5,
    splitter= 'random'
)

model.fit(df.drop('price',axis=1),df['price'])

joblib.dump(model,'./car_model.pkl')