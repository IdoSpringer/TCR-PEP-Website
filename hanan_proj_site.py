from flask import Flask, render_template, send_file, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import Prediction
import pandas as pd
from prediction import ae_predict, lstm_predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'd9c008b8a2ec6c4cd1371bc27174a889'
app.config['UPLOAD_FOLDER'] = "upload"
# app.config['baseURI'] = "http://peptibase.cs.biu.ac.il/Tcell_predictor/"

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

                # dict_results = Prediction.main(file_path)
                dict_results = ae_predict.main(file_path)

                os.remove(file_path)

                df = pd.DataFrame.from_dict(dict_results, orient='index')
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'temp_results.csv'))
                return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename='temp_results.csv')
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
    return send_from_directory(directory="static", filename="input-try.txt")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8085))
    app.run(host='0.0.0.0', port=port, debug=True)
    # app.run(debug=True)
