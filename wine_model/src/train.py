import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

df=pd.read_csv('../datasets/wine_train_data.csv')

model=DecisionTreeClassifier(
    random_state=42,
    criterion='entropy',
    max_depth= 20,
    max_features= None,
    min_samples_leaf= 1,
    min_samples_split=2,
    splitter='random'
)

model.fit(df.drop('quality',axis=1),df['quality'])

joblib.dump(model,'./wine_model.pkl')