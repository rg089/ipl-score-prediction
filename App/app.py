import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for

app = Flask("ipl")

@app.route("/", methods=["POST", "GET"])
def home():    
    if request.method == "POST":
        #user = request.form["nm"] 
        #psw= request.form["pwd"] 
        #lower, upper = predict([])
        return render_template("prediction.html")
    else:
        return render_template("index.html")


def predict(data):
    f = open("model.pkl", "rb")
    l = []
    for i in range(2):
        l.append(pickle.load(f))

    model, ohe = l
    l1 = ohe.transform(data[:3].reshape(1, -1))
    l2 = data[3:].reshape(1,-1).astype("float")
    X = np.concatenate((l1, l2), axis=1)

    print (int(model.predict(X)[0]))

# inputs = []
# data = np.array(inputs)
# predict(data)

if __name__ == "__main__":
    app.run(debug=True)