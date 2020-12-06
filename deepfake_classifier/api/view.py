from flask import Blueprint
from flask import request
from flask.json import jsonify
from werkzeug.utils import secure_filename

from deepfake_classifier.classifier.predict_folder import predict
from deepfake_classifier.converter.video_converter import web_to_mp4

import os

api = Blueprint('api', __name__)

ALLOWED_EXTENSIONS = {'.mp4'}
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'tmp')


@api.route('', methods=['POST'])
def detect_deepfake():
    output = {"status": ""}
    if 'file' not in request.files:
        output["status"] = "File not found in request"
        return jsonify(output)

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    print(file_path)
    if file.filename == '':
        output["status"] = "No file selected"
        return jsonify(output)

    elif file.filename == 'blob':
        file.save(file_path)
        if web_to_mp4(file_path, file_path + '.mp4') == 0:
            output["status"] = "Error converting file"
            return jsonify(output)
        else:
            os.remove(file_path)
            file_path += '.mp4'

    else:
        file_ext = os.path.splitext(secure_filename(file.filename))[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            output["status"] = file_ext + " is not supported"
            return jsonify(output)
        file.save(file_path)


    predict_result = predict(file_path)
    os.remove(file_path)

    print('Probability '+ str(predict_result))
    if predict_result == 100:
        output["status"] = "Not enough computational power. Try to upload videos with only one face"
        return jsonify(output)

    elif predict_result == 50:
        output["status"] = "No faces detected"
        return jsonify(output)

    elif predict_result == 200:
        output["status"] = "Unexpected error while predicting"
        return jsonify(output)

    elif predict_result > 0.90:
        output["status"] = "This video contains deepfake. Probability = " + str(predict_result*100)[:4] + "%"
        return jsonify(output)

    elif predict_result > 0.75:
        output["status"] = "This video may contain deepfake. Probability = " + str(predict_result*100)[:4] + "%"
        return jsonify(output)

    elif predict_result > 0.50:
        output["status"] = "This video is unlikely to contain deepfake. Probability = " + str(predict_result*100)[:4] + "%"
        return jsonify(output)

    elif predict_result > 0.30:
        output["status"] = "This video may not contain deepfake. Probability = " + str(predict_result*100)[:4] + "%"
        return jsonify(output)

    else:
        output["status"] = "This video does not contain deepfake. Probability = " + str(predict_result*100)[:4] + "%"
        return jsonify(output)
