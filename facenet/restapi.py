from flask import Flask, json, Response, request,jsonify
from os import path, getcwd
import face
import cv2
import numpy as np
import base64
import io
import ast
from flasgger import swag_from, Swagger

# INIT
app = Flask(__name__)
app.config['file_allowed'] = ['image/png', 'image/jpeg']
app.config['storage'] = path.join(getcwd(), './storage')
#app.face = Face(app)
app.config['SWAGGER'] = {
    'title': 'Face API',
    'uiversion': 3
}
swagger = Swagger(app)

# HTTP RESULTS

def http_success_result(success_message, data, status=200, mimetype='application/json'):
    return Response(json.dumps({"result": {"message": success_message, "data": data}}), status=status, mimetype=mimetype)

def http_error_result(error_message, error_code, status=500, mimetype='application/json'):
    return Response(json.dumps({"result": {"message": error_message, "code": error_code}}), status=status, mimetype=mimetype)

# ---------------------------

# Add new person and train dataset
@app.route('/api/newface', methods=['POST'])
def getfaceembendding():
    content = request.json
    if not content['image']:
        return http_error_result("Image is required.", "1005")
    else:     
        base64_image = content['image']
        img = base64.b64decode(base64_image)
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)

        face_recognition = face.Recognition(min_face_size=20)
        test_face = face_recognition.add_identity(source,'test_1')
        return jsonify(results = test_face.embedding.tolist())

# Train face dataset
@app.route('/api/verification', methods=['POST'])
def verification():
    content = request.json
    base64_image = content['image']
    embedding = content['emb']
    testarray = ast.literal_eval(embedding)
    arr =np.array(testarray)
    img = base64.b64decode(base64_image)
    npimg = np.fromstring(img, dtype=np.uint8)
    source = cv2.imdecode(npimg, 1)
    
    face_recognition = face.Recognition(min_face_size=20)
    test_face = face_recognition.identify(source,arr)
    print(test_face)
    print('test face')
    return jsonify(results = test_face[0].dist)


# ==============| RUN APP |==============
if __name__ == '__main__':
    app.run(host="0:0:0:0",port="80")
