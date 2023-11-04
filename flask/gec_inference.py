import os
import json
import flask
import pickle
import logging

# Load the model
# Ensure that the model pickle file is copied to this location during image build.
# TODO: Should inject the actual model through configuration
# TODO: Should inject the model path through configuration
model_path = f'gec_cows_l2h_small.pkl'
logging.info("Model Path" + str(model_path))

if not os.path.exists(model_path):
    logging.error(f'Missing model file {model_path}')

gec_model = None
with open(model_path, 'rb') as f:  # open a text file
    gec_model = pickle.load(f) # serialize the list

logging.info('GEC model loaded')

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        #regressor
        status = 200
        logging.info("Status : 200")
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/inference', methods=['POST'])
def transformation():
    if gec_model is None:
        logging.error(f'Model not initialized')
        return flask.Response(response= json.dumps({"error":"Model not initialized"}), status=500, mimetype='application/json' )

    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    line = input_json['line']
    predictions, _ = gec_model.predict([line])

    # Transform predictions to JSON
    result = {
        'output': predictions
    }

    return flask.Response(response=json.dumps(result), status=200, mimetype='application/json')