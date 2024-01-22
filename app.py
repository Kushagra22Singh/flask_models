import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# import mysql.connector as c 

# con=c.connect(host='localhost',user='root',passwd='your_password',database='your_DB')
# cursor=con.cursor()

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    #print(float_features)
    features = [np.array(float_features)]
    prediction = model.predict(features)
    print(prediction[0])
    cls = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    prediction = cls[prediction[0]]

    # query="""insert into records() values(%s,%s,%s,%s,%s)"""    
    # val_1=(float_features[0], float_features[1],float_features[2], float_features[3],prediction)
    # cursor.execute(query,val_1)
    # con.commit()

    return render_template("index.html", prediction_text = "The predicted flower species is {}".format(prediction))


# if __name__ == "__main__":
#     flask_app.run(debug=True)
