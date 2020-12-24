import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import pickle

df = pd.read_csv("Data/deliveries.csv")

def preprocessing_train(df_processed):
    df_processed = df_processed.loc[df_processed["overs"]>=6]
    df_processed.drop(["match_id", "inning", "over", "ball", "total_runs", "player_dismissed"], axis=1, inplace=True)
    df_processed = df_processed[['batting_team', 'bowling_team', 'toss_decision', 'overs' ,'current_score', 'current_wickets', 'last_6_runs', 'last_6_wickets', 'final_score']]
    small_categories = ["batting_team", "bowling_team", "toss_decision"]
    encode = OneHotEncoder(handle_unknown="ignore", sparse=False)
    dftemp = pd.DataFrame(encode.fit_transform(df_processed[small_categories]) , index=df_processed.index)
    df_processed = pd.concat((dftemp , df_processed.drop(small_categories, axis=1)), axis=1)
    X = df_processed.iloc[:, :-1].to_numpy()
    y = df_processed.iloc[:, -1].to_numpy()
    return X, y, encode

X, y, ohe = preprocessing_train(df)
model = XGBRegressor(learning_rate = 0.1, max_depth= 2, n_estimators=400)

model.fit(X, y)
f = open("model.pkl", "wb")
for i in [model, ohe]:
    pickle.dump(i, f)
f.close()
