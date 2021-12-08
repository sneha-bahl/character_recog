import os
from flask import Flask, render_template, request
from predictor import predict

app = Flask(__name__)

APP_ROOT  = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        resized_image_name="resized_" + filename
        print(destination)
        file.save(destination)
        result_image_name = predict(filename)
    return render_template("result.html", result = result_image_name, original=resized_image_name)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
