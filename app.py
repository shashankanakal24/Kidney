from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/solution')
def solution():
    return render_template('solution.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/experienced')
def experienced():
    return render_template('experienced.html')

@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/intern')
def intern():
    return render_template('intern.html')

@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')




@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    # os.system("python main.py")
    os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080) #for AWS


