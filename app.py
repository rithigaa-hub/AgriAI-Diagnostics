import os

from flask import Flask,render_template,request,redirect,url_for

from werkzeug.utils import secure_filename


from config import *

from predict import predict_image

from database import (
    create_database,
    insert_prediction,
    get_history
)



app=Flask(__name__)

app.secret_key=SECRET_KEY


app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER



# Create folders

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


os.makedirs(
    "database",
    exist_ok=True
)


create_database()



# Home Page

@app.route("/")
def home():

    return render_template(
        "index.html"
    )



# Prediction

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    image=request.files["image"]


    filename=secure_filename(
        image.filename
    )


    filepath=os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    image.save(filepath)



    disease,confidence=predict_image(
        filepath
    )


    insert_prediction(
        filename,
        disease,
        confidence
    )



    return render_template(
        "result.html",
        image=filename,
        disease=disease,
        confidence=round(confidence,2)
    )



# History

@app.route("/history")
def history():

    data=get_history()


    return render_template(
        "history.html",
        records=data
    )



# About

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


from flask import send_from_directory


@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )

if __name__=="__main__":

    app.run(
        debug=True
    )
