Business needs

    Predicts quality of red wine based on its chemical parameters, such sugar,alchohol, sulfates and some more. I choose classification for this model because target column 'quality' have integers in it. 

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
    
    After running script in src folder will appear wine_model.pkl file

    For getting predictions execute:
        python predict.py

    After running script in datasets folder will appear predictions.csv dataset