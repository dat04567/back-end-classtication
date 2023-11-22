from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin    
from .util import *


def create_app():
    
    app = Flask(__name__)
    app.config["CORS_HEADERS"] = "Content-Type"
    CORS(app)
    @cross_origin(origin="*", headers=["Content-Type"])
    @app.route('/api/classify_image', methods=['POST'])
   
    def classify_image():
        load_saved_artifacts()
        image_data = request.get_json()['image_data']
        
        response = jsonify(classify_image_predict(image_data))  
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        return response
   
    return app



    


 

