import sys
sys.path.append('../')
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(f"====================== gec-simple-inference::In ============================")

import os
import json
import pickle

logger.info(f"====================== gec-simple-inference::Importing proj stuff ============================")

import Config
import tensorflow as tf
from model_utils import create_simple_model, infer_simple
import sagemaker_ssh_helper
sagemaker_ssh_helper.setup_and_start_ssh()

JSON_CONTENT_TYPE = 'application/json'


logger.info(f"====================== gec-simple-inference::Running stuff ============================")

def _load_model_from_run_3emhdbgu(weights_loc):
    logger.info(f"====================== gec-simple-inference::About to load model ============================")
    
    main_args = Config()
    train_config = main_args.bert_plain_models[0]
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False)

    train_config['optimizer'] = optimizer
    train_config['batch_size'] = 32
    train_config['train_layers'] = -1
    train_config['epochs'] = 10
    train_config['dropout_rate'] = 0.5
    train_config['load_from_pretrained'] = True
    train_config['export_weights'] = False
    train_config['weights_loc'] = weights_loc
    train_config['extra_metrics_to_wandb'] = False
    train_config['publish_weights_to_wandb'] = False
    train_config['use_weighted_loss'] = True
    train_config['enable_wandb'] = False
    train_config['pretrain_model'] = False
    model = create_simple_model(main_args, train_config, verbosity=0)
    tokenizer = AutoTokenizer.from_pretrained(train_config['model_name'])
    return  model, tokenizer

def model_fn(model_dir):
    logger.info(f"inside model_fn, model_dir= {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))
    weights_loc = f'{model_dir}/temp'
    if not os.path.exists(f'{weights_loc}/checkpoint'):
        logging.error(f'Missing model weights file file {weights_loc}/checkpoint')

    # Create the model and load weights
    # The model configuration needs to match the exact model that produced the weights.
    # If the model is retrained and new weights are used the the model architecture needs to change as well.
    model, tokenizer = _load_model_from_run_3emhdbgu()
    model.to(device)
    logging.info(f'GEC model loaded into device {device}')
    return model, tokenizer

def predict_fn(data, model_tokenizer):
    model, tokenizer = model_tokenizer
    logger.info(f'Got input Data: {data}')
    return infer_simple(model, tokenizer, data)

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info(f"serialized_input_data object: {serialized_input_data}")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        logger.info(f"input_data object: {input_data}")
        return [input_data['line']]
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)

def output_fn(prediction, content_type):
    logger.info(f"prediction object before: {prediction}, type: {type(prediction)}")

    # Transform predictions to JSON
    prediction_result = {
        'output': prediction[0]
    }

    logger.info(f"prediction_result object: {prediction_result}")
    return prediction_result
