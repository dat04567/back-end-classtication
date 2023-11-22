import joblib
import json
import numpy as np
import base64
import cv2
import joblib
from .wavelet import w2d
from scipy.special import expit

__class_name_to_number = {}
__class_number_to_name = {}

__parameters = {}


def mle(y, axis=1):
    return np.argmax(y, 1)


def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores)

def sigmoid(z):
    return expit(z)

def forward_propagation(X, parameters):
    # retrieve the parameters
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    
    # compute the activation of the hidden layer
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    
    # compute the activation of the output layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache


def predict(X,parameters ):
   A2 , _ = forward_propagation(X, parameters)
   return mle(A2.T)

def predict_proba(X, parameters):
   A2 , _ = forward_propagation(X, parameters)
   return softmax(A2.T)

def classify_image_predict(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(predict(final, __parameters)[0]),
            'class_probability': np.around(predict_proba(final, __parameters)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    with open("./my_app/artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    global __parameters 
    
    if not __parameters :
        with open('./my_app/model/parameters.json', 'rb') as f:
            model = json.load(f)
        for key, value in model.items():
            __parameters[key] = np.array(value)
            
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./my_app/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./my_app/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image_predict(None, "./test/messi.jpeg"))
