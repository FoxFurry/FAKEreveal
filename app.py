from deepfake_classifier import app
from deepfake_classifier.classifier.predict_folder import initialize

if __name__ == '__main__':
    initialize()
    app.run()
