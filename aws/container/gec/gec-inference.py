import logging
import os
import torch
import json
import pickle

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    logger.info(f"inside model_fn, model_dir= {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    model_loc = f'{model_dir}/gec_cows_l2h_small.pkl'
    if not os.path.exists(model_loc):
        logging.error(f'Missing model file {model_loc}')

    model = None
    with open(model_loc, 'rb') as f:  # open a text file
        model = pickle.load(f) # serialize the list
        model.to(device)

    logging.info(f'GEC model loaded into device {device}')
    return model

def predict_fn(data, model):
    logger.info(f'Got input Data: {data}')
    prediction, _ = model.predict([data])

    return prediction

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info(f"serialized_input_data object: {serialized_input_data}")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        logger.info(f"input_data object: {input_data}")
        return input_data['line']
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)

def output_fn(prediction, content_type):
    logger.info(f"prediction object before: {prediction}, type: {type(prediction)}")

    # Transform predictions to JSON
    prediction_result = {
        'output': prediction
    }

    logger.info(f"prediction_result object: {prediction_result}")
    return prediction_result
