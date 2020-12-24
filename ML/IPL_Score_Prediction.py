import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import pickle

df = pd.read_csv("deliveries.csv")

df1 = pd.read_csv("matches.csv")

def preprocessing_train(df, df1, n):
    df.replace({"Delhi Daredevils": "Delhi Capitals"}, inplace=True)
    df1 = df1.iloc[:, [0, 7, 14]]
    df1.rename(columns={"id":"match_id"}, inplace=True)
    d = {'Feroz Shah Kotla Ground':'Feroz Shah Kotla', 'M Chinnaswamy Stadium':'M. Chinnaswamy Stadium', 'M. A. Chidambaram Stadium':'MA Chidambaram Stadium', 'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association Stadium, Mohali', 'Rajiv Gandhi International Stadium, Uppal':'Rajiv Gandhi Intl. Cricket Stadium', 'MA Chidambaram Stadium, Chepauk':'MA Chidambaram Stadium'}
    df1["venue"].replace(d, inplace=True)
    df_processed = pd.merge(df.iloc[:, [0,1,2,3,4,5,17,18]][df["inning"]==1], df1, on="match_id")
    df_processed["over"]-=1
    current_teams = ['Sunrisers Hyderabad', 'Royal Challengers Bangalore','Mumbai Indians','Kolkata Knight Riders', 'Kings XI Punjab','Chennai Super Kings', 'Rajasthan Royals','Delhi Capitals']
    df_processed = df_processed[(df_processed["batting_team"].isin(current_teams)) & (df_processed["bowling_team"].isin(current_teams))]
    df_processed["overs"] = df_processed["over"]+df_processed["ball"]*0.1
    runs = df_processed.groupby("match_id")["total_runs"].sum()
    df_processed["final_score"] = df_processed["match_id"].apply(lambda x: runs.loc[x])
    df_processed["current_score"] = df_processed.groupby("match_id")["total_runs"].cumsum()
    df_processed["last_5_runs"] = df_processed.groupby("match_id")["total_runs"].rolling(n*6).sum().values
    df_processed.fillna({"player_dismissed":0}, inplace=True)
    df_processed.loc[df_processed["player_dismissed"]!=0,"player_dismissed"]=1
    df_processed["player_dismissed"] = pd.to_numeric(df_processed["player_dismissed"])
    df_processed["last_5_wickets"] = df_processed.groupby("match_id")["player_dismissed"].rolling(n*6).sum().values
    df_processed["current_wickets"] = df_processed.groupby("match_id")["player_dismissed"].cumsum()
    df_processed = df_processed[df_processed["overs"]>=n]
    df_processed.drop(["match_id", "inning", "over", "ball", "total_runs", "player_dismissed"], axis=1, inplace=True)
    df_processed = df_processed[['batting_team', 'bowling_team', "toss_decision", "venue", 'overs' ,'current_score', 'current_wickets', 'last_5_runs', 'last_5_wickets', 'final_score']]
    le = LabelEncoder()
    df_processed["venue"] = le.fit_transform(df_processed["venue"])
    small_categories = ["batting_team", "bowling_team", "toss_decision"]
    encode = OneHotEncoder(handle_unknown="ignore", sparse=False)
    dftemp = pd.DataFrame(encode.fit_transform(df_processed[small_categories]) , index=df_processed.index)
    df_processed = pd.concat((dftemp , df_processed.drop(small_categories, axis=1)), axis=1)
    X = df_processed.iloc[:, :-1]
    y = df_processed.iloc[:, -1]
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X)
    return X1, y, encode, le, scaler

X, y, ohe, le, scaler= preprocessing_train(df, df1, 6)
model = Lasso(alpha=0.2, max_iter=1000)

#print (-1*cross_val_score(model, X, y, scoring = "neg_mean_squared_error", cv=3).mean())

model.fit(X, y)
f = open("model.pkl", "wb")
for i in [model, ohe, le, scaler]:
    pickle.dump(i, f)
f.close()
