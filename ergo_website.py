from flask import Flask, render_template, send_file, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from prediction import ae_predict, lstm_predict
import csv
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = b"\x82\x95\xef\x02\x1c\x08bz'\xc40\x1a\xed4\xdf\xe0"

app.config['UPLOAD_FOLDER'] = "upload"
app.config['baseURI'] = "http://peptibase.cs.biu.ac.il/ERGO/"

@app.route("/")
@app.route("/home/", methods=['GET', 'POST'])
def home():
    error_message = False

    if request.method == 'POST':
        try:
            # check if the post request has the file part
            if len(request.files) == 0:
                flash('No file part')
                return redirect(request.url)
            # get first file
            file = request.files.values().__next__()
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                model = request.form['model']
                if model == 'ae':
                    results = ae_predict.main(file_path)
                elif model == 'lstm':
                    results = lstm_predict.main(file_path)

                os.remove(file_path)

                with open(app.config['UPLOAD_FOLDER'] + '/results.csv', 'w') as file:
                    writer = csv.writer(file)
                    for result in results:
                        writer.writerow(result)
                return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename='results.csv')
        except:
            error_message = True

    return render_template("home.html", error_message=error_message)


@app.route("/help/")
def help():
    return render_template("help.html")


@app.route("/example/")
def example():
    return render_template("example.html")


@app.route("/download_example/")
def download_example():
    return send_from_directory(directory="static", filename="pairs_example.csv")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8085))
    app.run(host='0.0.0.0', port=port, debug=True)
    # app.run(debug=True)
