import os
from flask import Flask, request, render_template
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import sqlite3


conn = sqlite3.connect('gallery.db')
c = conn.cursor()


app = Flask(__name__)
uFolder = "static"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
model = tensorflow.keras.models.load_model('model/keras_model.h5', compile=False)
app.config['SECRET_KEY'] = 'Hello'


def classify(img_path):
    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path)
    image.convert('1')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    classLabels = ["AlfaRomeo4C", "Non"]
    return (classLabels[np.argmax(prediction)])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        img_file = request.files["image"]
        if img_file.filename =='':
            return render_template("index.html", p=0, image_loc=None)
        if img_file:
                img_location = os.path.join(
                    uFolder,
                    img_file.filename
                )
                img_file.save(img_location)
                result = classify(img_location)

                con = sqlite3.connect('gallery.db')
                cur = con.cursor()
                cur.execute("INSERT INTO my_gallery2 (prediction,image_path) VALUES (?, ?)", (result, img_location))
                con.commit()
                print(result)
                return render_template("index.html", p=result, image_loc=img_file.filename)

    return render_template("index.html", p=0, image_loc=None)


@app.route('/list')
def list():
    con = sqlite3.connect('gallery.db')
    con.row_factory = sqlite3.Row

    cur = con.cursor()
    cur.execute("select * from my_gallery2")

    rows = cur.fetchall()
    return render_template("list.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True)
