import json
import utils
import settings
import os
from middleware import model_predict
import csv

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    jsonify,
)

router = Blueprint("app_router", __name__, template_folder="templates")

# GET method in order to render template for first time.
@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code.
    """
    return render_template("index.html")


@router.route("/", methods=["POST"])
def upload_image():
    """
    Function used in our frontend so we can upload and show an image.
    When it receives an image from the UI, it also calls our ML model to
    get and display the predictions.
    """
    # No file received, show basic UI
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    # File received but no filename is provided, show basic UI
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)

    # File received and it's an image, we must show it and get predictions
    if file and utils.allowed_file(file.filename):
        unique_file_name = utils.get_file_hash(file)
        path = os.path.join(settings.UPLOAD_FOLDER, unique_file_name)
        if not os.path.exists(path):
            file.save(path)

        context = {
            "prediction": None,
            "score": None,
            "filename": None,
        }
        results = model_predict(unique_file_name)
        context.update({"prediction": results[0],
                        "score": results[1],
                        "filename": unique_file_name})

        return render_template("index.html", filename=unique_file_name, context=context)

    # File received and but it isn't an image
    else:
        flash("Allowed image types are -> png, jpg, jpeg, gif")
        return redirect(request.url)


@router.route("/display/<filename>")
def display_image(filename):
    """
    Display uploaded image in our UI.
    """
    return redirect(
        url_for("static", filename="uploads/" + filename), code=301
    )


@router.route("/predict", methods=["POST"])
def predict():
    if 'file' in request.files:
        file = request.files["file"]

        if file and utils.allowed_file(file.filename):
            unique_file_name = utils.get_file_hash(file)
            file.save(settings.UPLOAD_FOLDER + unique_file_name)
            results = model_predict(unique_file_name)

            rpse = {"success": True, "prediction": results[0], "score": results[1]}
            return jsonify(rpse)

    rpse = {"success": False, "prediction": None, "score": None}
    return jsonify(rpse), 400

@router.route("/feedback", methods=["GET", "POST"])
def feedback():

    # Get reported predictions from `report` key
    report1 = request.form.get("report1")
    report2 = request.form.get("report2")
    report3 = request.form.get("report3")
    fb_path =  settings.FEEDBACK_FILEPATH2

    if not os.path.exists(fb_path):
        with open(fb_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Prediction', 'Score'])
            writer.writerow([str(report1),str(report2),str(report3)])
        flash('Thank you for your feedback!')
    else:
        with open(fb_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([str(report1),str(report2),str(report3)])
        flash('Thank you for your feedback!')
        
    return render_template("index.html")
