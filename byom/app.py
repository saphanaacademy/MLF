from base64 import b64encode
import os
from grpc.beta import implementations
import tensorflow as tf
import skimage.io
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import json
import requests
from flask import Flask, render_template, send_from_directory, globals, request, Response
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = '5a?L5Y%"kft~;7L/4onQ2(Kwh>;nJK5U7xZI>#R+.>%:BhF?/j1dAs;(k^A1}Xd'
socketio = SocketIO(app)

# get MLF service details
service = (json.loads(os.getenv('VCAP_SERVICES', '')))[str(os.getenv('MLF_SERVICE', ''))][0]
client_id = str(service['credentials']['clientid'])
client_secret = str(service['credentials']['clientsecret'])
authentication_url = str(service['credentials']['url']) + "/oauth/token"
deployment_url = str(service['credentials']['serviceurls']['DEPLOYMENT_API_URL']) + "/api/v2/modelServers"

def get_access_token():
    # obtain an OAuth2 access token
    query_string = {"grant_type": "client_credentials"}
    auth = b64encode(b"" + client_id + ":" + client_secret).decode("ascii")
    headers = {
        'Cache-Control': "no-cache",
        'Authorization': "Basic %s" % auth
    }
    res = requests.request("GET", authentication_url, headers=headers, params=query_string)
    return 'Bearer ' + json.loads(res.text)['access_token']

def metadata_transformer(metadata):
    # transorm metadata
    additions = []
    token = get_access_token()
    additions.append(('authorization', token))
    return tuple(metadata) + tuple(additions)

# path in route should match deployed model name
@app.route('/mnist', methods=['POST'])
def do_inference():
    # get deployed model details
    model_name = request.path[1:]
    query_string = {"modelName": model_name}
    headers = {
        'Authorization': get_access_token(),
        'Cache-Control': "no-cache"
    }
    res = requests.request("GET", deployment_url, headers=headers, params=query_string)
    model_info = json.loads(res.text)
    # check model is available
    if int(model_info["count"]) < 1:
        return Response('404 Not Found: Model ' + model_name + ' is unavailable.', status=404)
    else:
        # get details for the latest model version
        latest_version = [0, 0]
        for index, model in enumerate(model_info["modelServers"]):
            if int(model["specs"]["models"][0]["modelVersion"]) > latest_version[0]:
                latest_version = [int(model["specs"]["models"][0]["modelVersion"]), index]
        model_host = model_info["modelServers"][latest_version[1]]["endpoints"][0]
        credentials = implementations.ssl_channel_credentials(root_certificates = str(model_host["caCrt"]))
        channel = implementations.secure_channel(str(model_host["host"]), int(model_host["port"]),  credentials)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel, metadata_transformer=metadata_transformer)
        # check file was uploaded
        if not 'file' in globals.request.files:
            return Response('404 Not Found: File not provided.', status=404)
        else:
            # perform inference on the file 
            data = skimage.io.imread(globals.request.files['file'])
            req = predict_pb2.PredictRequest()
            req.model_spec.name = model_name
            req.model_spec.signature_name = 'predict_images'
            req.inputs["images"].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[1, data.size], dtype="float32"))
            res = stub.Predict(req, 150)
            # convert scores to JSON
            res = str(res).split('}')[3].split('\n')
            res.pop(11)
            res.pop(0)
            scores = {}
            for i, estimate in enumerate(res):
                scores[str(i)] = float(estimate[14:])
    return Response(json.dumps(scores), status=200, mimetype='application/json')

port = os.getenv('PORT', 5000)
if __name__ == '__main__':
    app.debug = not os.getenv('PORT')
    socketio.run(app, host='0.0.0.0', port=int(port))
