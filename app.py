from flask import Flask, request, jsonify
import util as util
from model.NNClassifier import NNClassifier
from flask_cors import CORS
app = Flask(__name__)

CORS(app, origins='https://classtication.web.app')

@app.route('/api/classify_image', methods=['POST'])
def classify_image():
    image_data = request.get_json()['image_data']

    response = jsonify(util.classify_image(image_data))  
    
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
 

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run()