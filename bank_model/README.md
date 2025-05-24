Business needs

    This model allow you to check your credit worthiness for one credit. It works this way, you pass parameters of credit you`d like to issue, and model give you level of credit worthiness, basicaly this level is chance of approving credit. Dataset contains credit data somewhere from USA, so interest rates are kinda small (3-5%) comparing to ukrainian rates of interest (13-17%). I have used classification because target column credit worthiness have two classes l1 and l2

Requirements

    python 3.12
    numpy
    pandas
    sklearn
    matplotlib
    joblib

Running: 
    For building model execute:
        python train.py
    
    After running script in src folder will appear bank_model.pkl file

    For getting predictions execute:
        python predict.py

    After running script in datasets folder will appear predictions.csv dataset