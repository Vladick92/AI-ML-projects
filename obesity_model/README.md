Business needs

    There are two models that allow you to calculate BMI and Label(like underweight, obese, etc). For calculating BMI i have used regression and for Labels classification. Thats just two separate models that were trained on same dataset. Reason why i choose to do two models, is that i cant deside which column choose as target, like BMI or Label. And i did little research about BMI, even though its simple formula, you can see it in notebook.ipynb

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
    
    After running script in src folder will appear two models bmi_model.pkl and label_model.pkl

    For getting predictions execute:
        python predict.py

    After running script in datasets folder will appear predictions.csv dataset