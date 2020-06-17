from flask import Flask, render_template, send_file, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from ERGO_files import ERGO_predict
import csv
import argparse

app = Flask(__name__)
app.config['SECRET_KEY'] = b"\x82\x95\xef\x02\x1c\x08bz'\xc40\x1a\xed4\xdf\xe0"

app.config['UPLOAD_FOLDER'] = "upload"
app.config['baseURI'] = "http://tcr.cs.biu.ac.il"


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
                test_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(test_file_path)
                model_type = request.form['model_type']
                dataset = request.form['dataset']
                parser = argparse.ArgumentParser()
                parser.add_argument("--model_type")
                parser.add_argument("--dataset")
                parser.add_argument("--device", default='cpu')
                parser.add_argument("--ae_file", default='auto')
                parser.add_argument("--model_file", default='auto')
                parser.add_argument("--test_data_file")
                args = parser.parse_args()
                args.model_type = model_type
                args.dataset = dataset
                args.test_data_file = test_file_path
                tcrs, peps, preds = ERGO_predict.predict(args)
                os.remove(test_file_path)
                with open(app.config['UPLOAD_FOLDER'] + '/results.csv', 'w') as file:
                    writer = csv.writer(file)
                    for tcr, pep, pred in zip(tcrs, peps, preds):
                        writer.writerow([tcr, pep, pred])
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
    port = int(os.environ.get('PORT', 8086))
    app.run(host='0.0.0.0', port=port, debug=True)
    # app.run(host='127.0.0.1', port=port, debug=True)
    # app.run(debug=True)
