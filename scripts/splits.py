import pickle
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd

rsk = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=6,
    random_state=42
)

df = pd.read_csv('df_final.csv', sep=',')

X = df.drop(columns=['last_loan_status'])
y = df['last_loan_status']

splits = list(rsk.split(X, y))

with open("splts.pkl", "wb") as f:
   pickle.dump(splits, f)