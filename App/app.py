import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for

app = Flask("ipl")

@app.route("/", methods=["POST", "GET"])
def home():    
    if request.method == "POST":
        team1 = request.form["batting"] 
        team2 = request.form["bowling"]
        t1 = map_team(team1); t2 = map_team(team2)
        toss = request.form["toss"]
        if toss=="bat":
            s = "after opting to bat"
        else:
            s = "after being put to bat"
        overs = float(request.form["overs"])
        score = int(request.form["current_score"])
        wickets = int(request.form["current_wickets"])
        score_6 = int(request.form["last_6_score"])
        wickets_6 = int(request.form["last_6_wickets"])
        data = [team1, team2, toss, overs, score, wickets, score_6, wickets_6]
        score = predict(np.array(data))
        return render_template("prediction.html", t1 = t1, t2 = t2, s=s, score=score, data=request.form)
    else:
        return render_template("index.html")

def map_team(s):
    if s=="Kings XI Punjab":
        return "kxip"
    elif s=="Sunrisers Hyderabad":
        return "srh"
    return "".join(map(lambda x: x[0].lower(), s.split()))




def predict(data):
    f = open("model.pkl", "rb")
    l = []
    for i in range(2):
        l.append(pickle.load(f))

    model, ohe = l
    l1 = ohe.transform(data[:3].reshape(1, -1))
    l2 = data[3:].reshape(1,-1).astype("float")
    X = np.concatenate((l1, l2), axis=1)

    prediction = int(model.predict(X)[0])

    return prediction

# inputs = []
# data = np.array(inputs)
# predict(data)

if __name__ == "__main__":
    app.run(debug=True)