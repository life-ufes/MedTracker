import pandas as pd
from sklearn.metrics import classification_report
import joblib



FILE = "/Teste4_output_4/airtag/rssi_median_filtered_airtag_mqtt.csv"
df = pd.read_csv(FILE, sep=",")

columns = ['time_window_start', 'time_window_end']

df.drop(columns, axis=1, inplace=True)
df.fillna(-100, inplace=True)

y = df['location']

x = df.drop(columns=['location'])

model = joblib.load("model.joblib")

y_pred = model.predict(x)

print(classification_report(y, y_pred))