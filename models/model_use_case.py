import pandas as pd
import numpy as np
from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
import joblib



FILE = "Teste4_output_4/airtag/rssi_median_filtered_airtag_mqtt.csv"
MODEL_TO_SAVE = 'RF' # Model that will be used for prediction can be KNN, RF or LR


def calculate_accuracy_stratified_cv(pipelines, X, y, n_splits=5):
    cross_val = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for name, pipeline in pipelines.items():
        accuracies = []
        for train_index, test_index in cross_val.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            accuracies.append(accuracy)
        
        print(f'{name}:')
        print(f'Média das acurácias: {np.mean(accuracies)}\n')




df = pd.read_csv(FILE, sep=",")

columns = ['time_window_start', 'time_window_end']

df.drop(columns, axis=1, inplace=True)
df.fillna(-100, inplace=True)

y = df['location']

x = df.drop(columns=['location'])


lr_pipeline = Pipeline(steps=[('Normalizacao', StandardScaler()), ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000, solver='saga'))])
rf_pipeline = Pipeline(steps=[('Normalizacao', MinMaxScaler()), ('RandomForest', RandomForestClassifier(random_state=42))])
knn_pipeline = Pipeline(steps=[('Normalizacao', MinMaxScaler()), ('KNN', KNeighborsClassifier(n_neighbors=5))])

pipelines = {
    'LR': lr_pipeline,
    'RF': rf_pipeline,
    'KNN': knn_pipeline,
}

calculate_accuracy_stratified_cv(pipelines, x, y)

model = pipelines[MODEL_TO_SAVE]

model.fit(x, y)

joblib.dump(model, 'model.joblib')