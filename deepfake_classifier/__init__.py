from flask import Flask
from flask_cors import CORS
from deepfake_classifier.api.view import api
app = Flask(__name__)
CORS(app)

app.register_blueprint(api, url_prefix='/api')

