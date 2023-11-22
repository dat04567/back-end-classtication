from flask import Flask, request, jsonify
from flask_cors import CORS
from .util import *
from  .NNClassifier import NNClassifier

def create_app():
    
    app = Flask(__name__)
    CORS(app, origins='https://classtication.web.app')
    
    @app.route('/api/classify_image', methods=['POST'])
   
    def classify_image():
        load_saved_artifacts()
        image_data = request.get_json()['image_data']
        
        response = jsonify(classify_image_predict(image_data))  
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        return response
   
    return app



    


 

