import pickle
import wandb
import tensorflow as tf
from transformers import TFBertModel, AutoTokenizer
from tensorflow.keras.backend import sparse_categorical_crossentropy
from tensorflow.keras.layers import Flatten
import tarfile
import numpy as np
import re

def download_simple_model(run_path, weights_filename):
    api = wandb.Api()
    run = api.run(path=run_path)
    print(f'Found {run} to load artifact from')
    weights_file = run.file(weights_filename)
    print(f'Downloading {weights_filename}')
    weights_file.download(replace=True, root='./downloads/')
    with tarfile.open(f'downloads/{weights_filename}', "r:gz") as tar:
        tar.extractall()


def create_simple_model(main_args, exp_config, verbosity=1):
    """
      Creates a new model.

      If load_from_pretrained parameter is True then weights are loaded from weights_loc
      else the model is trained and weights are saved to weights_loc.
    """
    # Load state and data for config
    with open(f'{main_args.PROCESSED_DATA_FOLDER}/{exp_config["train_data_file"]}', "rb") as in_file:
        training_data = pickle.load(in_file)

    NUM_ORIG_CLASSES = training_data['NUM_ORIG_CLASSES']
    NUM_NER_CLASSES = training_data['NUM_NER_CLASSES']
    NUM_TOTAL_CLASSES = training_data['NUM_TOTAL_CLASSES']
    numNerClasses = NUM_TOTAL_CLASSES
    numSentences = training_data['numSentences']
    max_length = exp_config['max_length']
    train_all = training_data['train_all']
    [bert_inputs_train_k, labels_train_k] = train_all
    # Calculate class weights for loss function
    # freq_table = np.unique(labels_train_k.flatten(), return_counts=True)
    class_weights = {
        0: 0.3293928,
        1: 0.65469572,
        2: 0.01591148,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    # class_weights = tf.compat.v2.constant([*freq_table[1][0:3], 0, 0, 0 0])

    test_all = training_data['test_all']
    [bert_inputs_test_k, labels_test_k] = test_all
    cat_list = training_data['nerClassesTag'].categories.tolist()
    orig_cat_list = cat_list[0:NUM_ORIG_CLASSES]


    def masked(y_true, y_pred, classes_to_include):
        y_label = tf.reshape(tf.keras.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])
        mask = (y_label < classes_to_include) # Filter for first classes_to_include classes
        y_label_masked = tf.boolean_mask(y_label, mask)
        y_predicted = tf.math.argmax(input = tf.reshape(tf.keras.layers.Flatten()(tf.cast(y_pred, tf.float64)),\
                                                        [-1, numNerClasses]), axis=1)
        y_predicted_masked = tf.boolean_mask(y_predicted, mask)
        # Convert to actual labels and return
        return y_label_masked.map(lambda v: cat_list[v]), y_predicted_masked.map(lambda v: cat_list[v])

    def recall_sans_other(y_true, y_pred):
        y_label_masked, y_predicted_masked = masked(y_true, y_pred, NUM_ORIG_CLASSES - 1)
        return
        return true_positives / (possible_positives + K.epsilon())


    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras


    def specificity(y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())


    def negative_predictive_value(y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        return tn / (tn + fn + K.epsilon())


    def f1(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def custom_loss_all_orig_lables(y_true, y_pred):
        """
        calculate loss function explicitly, filtering out 'extra inserted labels'

        y_true: Shape: (batch x (max_length + 1) )
        y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens )

        returns:  cost
        """
        #get labels and predictions
        y_label = tf.reshape(Flatten()(tf.cast(y_true, tf.int32)),[-1])
        mask = (y_label < NUM_ORIG_CLASSES)   # This mask is used to remove all tokens that
                                              #  do not correspond to the original base text
                                              #  and the other token.
        y_label_masked = tf.boolean_mask(y_label, mask)  # mask the labels
        y_flat_pred = tf.reshape(Flatten()(tf.cast(y_pred, tf.float32)),[-1, numNerClasses])
        y_flat_pred_masked = tf.boolean_mask(y_flat_pred, mask) # mask the predictions
        return tf.reduce_mean(sparse_categorical_crossentropy(y_label_masked, y_flat_pred_masked,from_logits=False ))

    def custom_loss_other_ignored(y_true, y_pred):
        """
        calculate loss function explicitly, filtering out 'extra inserted labels'

        y_true: Shape: (batch x (max_length + 1) )
        y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens )

        returns:  cost
        """
        #get labels and predictions
        y_label = tf.reshape(Flatten()(tf.cast(y_true, tf.int32)),[-1])
        mask = (y_label < NUM_ORIG_CLASSES - 1)   # This mask is used to remove all tokens that
                                                  #  do not correspond to the original base text
                                                  #  and the other token.
        y_label_masked = tf.boolean_mask(y_label, mask)  # mask the labels
        y_flat_pred = tf.reshape(Flatten()(tf.cast(y_pred, tf.float32)),[-1, numNerClasses])
        y_flat_pred_masked = tf.boolean_mask(y_flat_pred, mask) # mask the predictions
        return tf.reduce_mean(sparse_categorical_crossentropy(y_label_masked, y_flat_pred_masked,from_logits=False ))

    def custom_loss_weighted(y_true, y_pred):
        """
        calculate loss function explicitly, filtering out 'extra inserted labels'

        y_true: Shape: (batch x (max_length + 1) )
        y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens )

        returns:  cost
        """
        # class_weights
        #get labels and predictions
        y_label = tf.reshape(Flatten()(tf.cast(y_true, tf.int32)),[-1])
        mask = (y_label < NUM_ORIG_CLASSES) # This mask is used to remove all tokens that
                                            #  do not correspond to the original base text
        y_label_masked = tf.boolean_mask(y_label, mask)  # mask the labels
        y_flat_pred = tf.reshape(Flatten()(tf.cast(y_pred, tf.float32)),[-1, numNerClasses])
        y_flat_pred_masked = tf.boolean_mask(y_flat_pred, mask) # mask the predictions
        return tf.reduce_mean(sparse_categorical_crossentropy(y_label_masked, y_flat_pred_masked,from_logits=False ))

    def accuracy_all_classes(y_true, y_pred):
        """
        calculate loss dfunction filtering out also the newly inserted labels

        y_true: Shape: (batch x (max_length) )
        y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens )

        returns: accuracy
        """

        #get labels and predictions
        y_label = tf.reshape(tf.keras.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])
        mask = (y_label < NUM_ORIG_CLASSES)
        y_label_masked = tf.boolean_mask(y_label, mask)
        y_predicted = tf.math.argmax(input = tf.reshape(tf.keras.layers.Flatten()(tf.cast(y_pred, tf.float64)),\
                                                        [-1, numNerClasses]), axis=1)
        y_predicted_masked = tf.boolean_mask(y_predicted, mask)
        return tf.reduce_mean(tf.cast(tf.equal(y_predicted_masked,y_label_masked) , dtype=tf.float64))

    def accuracy_sans_other_class(y_true, y_pred):
        """
        calculate loss dfunction explicitly filtering out also the 'Other'- labels

        y_true: Shape: (batch x (max_length) )
        y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens )

        returns: accuracy
        """

        #get labels and predictions
        y_label = tf.reshape(tf.keras.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])
        mask = (y_label < NUM_ORIG_CLASSES - 1)
        y_label_masked = tf.boolean_mask(y_label, mask)
        y_predicted = tf.math.argmax(input = tf.reshape(tf.keras.layers.Flatten()(tf.cast(y_pred, tf.float64)),\
                                                        [-1, numNerClasses]), axis=1)
        y_predicted_masked = tf.boolean_mask(y_predicted, mask)
        return tf.reduce_mean(tf.cast(tf.equal(y_predicted_masked,y_label_masked) , dtype=tf.float64))

    def export_metrics(model):
        result = model.predict(
            [bert_inputs_test_k[0], bert_inputs_test_k[1], bert_inputs_test_k[2]],
            batch_size=16
        )

        predictions_flat = [pred for preds in np.argmax(result, axis=2) for pred in preds]
        labels_flat = [label for labels in labels_test_k for label in labels]

        clean_preds = []
        clean_labels = []
        for pred, label in zip(predictions_flat, labels_flat):
            if label < NUM_ORIG_CLASSES:
                clean_preds.append(cat_list[pred])
                clean_labels.append(cat_list[label])

        wandb_truth = labels_test_k.reshape(labels_test_k.shape[0]*labels_test_k.shape[1], 1)
        wandb_preds = result.reshape(result.shape[0]*result.shape[1], result.shape[2])

        wandb.log({
            "roc":
            wandb.plot.roc_curve(
                wandb_truth,
                wandb_preds,
                labels=cat_list,
                title='Receiver Operating Characteristic Curve',
                classes_to_plot=list(range(0, NUM_ORIG_CLASSES))
            )
        })

        # Precision Recall
        wandb.log({
            "pr":
            wandb.plot.pr_curve(
                wandb_truth,
                wandb_preds,
                labels=cat_list,
                title='Precision-Recall Curve',
                classes_to_plot=list(range(0, NUM_ORIG_CLASSES))
            )
        })

        # Foc creating confusion matrix for just the 3 classes, we ignore
        # samples where either the truth or the predicted are not
        # one of the classes of interest.
        clean_preds = []
        clean_labels = []
        for pred, label in zip(predictions_flat, labels_flat):
            if label < NUM_ORIG_CLASSES and pred < NUM_ORIG_CLASSES:
                clean_preds.append(cat_list[pred])
                clean_labels.append(cat_list[label])

        wandb.log({
            "conf_mat_orig_classes": wandb.sklearn.plot_confusion_matrix(
                clean_labels, clean_preds, labels=orig_cat_list),
            "conf_mat_all_classes": wandb.sklearn.plot_confusion_matrix(
                labels_flat, predictions_flat, labels=cat_list),
        })

    def freeze_layers(bert_layer, train_layers):
        if not train_layers == -1:
            retrain_layers = []
            for retrain_layer_number in range(train_layers):
                layer_code = '_' + str(11 - retrain_layer_number)
                retrain_layers.append(layer_code)
            retrain_layers = set(retrain_layers)
            for w in bert_layer.weights:
                if not any([x in w.name for x in retrain_layers]):
                    w._trainable = False

    def ner_model(max_input_length, train_layers, optimizer, dropout_rate, use_weighted_loss):
        """
        Implementation of NER model

        variables:
            max_input_length: number of tokens (max_length + 1)
            train_layers: number of layers to be retrained
            optimizer: optimizer to be used

        returns: model
        """
        in_id = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name="segment_ids")


        bert_inputs = [in_id, in_mask, in_segment]

        # Note: Bert layer from Hugging Face returns two values: sequence ouput, and pooled output. Here, we only want
        # the former. (See https://huggingface.co/transformers/model_doc/bert.html#tfbertmodel)
        bert_layer = TFBertModel.from_pretrained(exp_config['model_name'])

        # Freeze layers, i.e. only train number of layers specified, starting from the top
        freeze_layers(bert_layer, train_layers)
        # End of freezing section

        bert_sequence = bert_layer(bert_inputs)[0]

        # Add a dropout layer for generalization.
        dense = tf.keras.layers.Dense(256, activation='relu', name='dense')(bert_sequence)
        dense = tf.keras.layers.Dropout(rate=dropout_rate)(dense)

        # Add the classification head as a dense layer
        pred = tf.keras.layers.Dense(numNerClasses, activation='softmax', name='ner')(dense)

        # Create, compile and return the model
        model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        model.compile(
            loss={"ner": sparse_categorical_crossentropy,},
            optimizer=optimizer,
            metrics=[accuracy_all_classes, accuracy_sans_other_class])
        if verbosity > 0:
            model.summary()
        return model, bert_layer

    enable_wandb = exp_config['enable_wandb']
    wandb_run_notes = exp_config['wandb_run_notes'] if 'wandb_run_notes' in exp_config else None

    if exp_config['pretrain_model']:
        # First pretrain the classification with BERT layers frozen
        pretraining_epochs = exp_config['pretraining_epochs']
        pretraining_optimizer = exp_config['pretraining_optimizer']
        pretraining_dropout = exp_config['pretraining_dropout']
        model, bert_layer = ner_model(
            max_length + 1,
            train_layers=0,
            optimizer=pretraining_optimizer,
            dropout_rate=pretraining_dropout,
            use_weighted_loss=exp_config['use_weighted_loss'])

        if enable_wandb:
            # Init a new wandb project
            wandb.init(
                config=exp_config,
                sync_tensorboard=True,
                project=exp_config['project_name'],
                mode='online',
                name=f'{exp_config["exp_name"]}_preTraining',
                notes=wandb_run_notes)

        model.fit(
            bert_inputs_train_k,
            {"ner": labels_train_k },
            validation_data=(bert_inputs_test_k, {"ner": labels_test_k }),
            epochs=pretraining_epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(histogram_freq=1)
            ],
            batch_size=exp_config['batch_size'] if 'batch_size' in exp_config else 16
        )
        # Then unfreeze all BERT layers and re-compile
        freeze_layers(bert_layer, -1)
        model.compile(
            loss={"ner": sparse_categorical_crossentropy,},
            optimizer=exp_config['optimizer'],
            metrics=[accuracy_all_classes, accuracy_sans_other_class])
    else:
        # Instantiate the model and train it.
        model, _ = ner_model(
            max_length + 1,
            train_layers=exp_config['train_layers'],
            optimizer=exp_config['optimizer'],
            dropout_rate=exp_config['dropout_rate'],
            use_weighted_loss=exp_config['use_weighted_loss'])

        # If load_from_pretrained is true then load the weights and return model.
        if exp_config['load_from_pretrained']:
            model.load_weights(exp_config['weights_loc'])
            return model

    weights_path = 'outputs/model_weights/'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        save_weights_only=True,
        verbose=1,
        monitor='val_accuracy_sans_other_class',
        save_best_only=True,
        mode='max')

    epochs = 5 # Default to 5 epochs.
    if 'epochs' in exp_config:
        epochs = exp_config['epochs']

    if enable_wandb:
        # Init a new wandb project
        wandb.init(
            config=exp_config,
            sync_tensorboard=True,
            project=exp_config['project_name'],
            mode='online',
            name=exp_config['exp_name'] if 'exp_name' in exp_config else None,
            notes=wandb_run_notes)

    model.fit(
        bert_inputs_train_k,
        {"ner": labels_train_k },
        validation_data=(bert_inputs_test_k, {"ner": labels_test_k }),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(histogram_freq=1), cp_callback
        ],
        batch_size=exp_config['batch_size'] if 'batch_size' in exp_config else 16
    )

    if enable_wandb:
        if 'extra_metrics_to_wandb' in exp_config and exp_config['extra_metrics_to_wandb']:
            print(f'Exporting PR and ROC to wandb')
            # Calculate additional metrics and upload to wandb
            export_metrics(model)

        if 'publish_weights_to_wandb' in exp_config and exp_config['publish_weights_to_wandb']:
            print(f'Uploading model weights to wandb')
            model.save_weights('temp/model_weights/', overwrite=True)
            tarball = 'model_weights.gz'
            with tarfile.open(tarball, "w:gz") as tar:
                tar.add('temp/model_weights/')
            wandb.save(tarball)

        wandb.finish()
    return model

def _word_to_class(sentenceList, preds, wordIndexTokenPosList):
    result = []
    catMap = {0: 'B-ga', 1: 'B-na', 2: 'O'}
    for sentenceId, sentence in enumerate(sentenceList):
        words = []
        for wordId, word in enumerate(sentence):
            catId = preds[sentenceId][wordIndexTokenPosList[sentenceId][wordId]]
            wordCat = catMap[catId] if catId in catMap else 'O'
            words.append({word: wordCat})
        result.append(words)
    return result

def infer_simple(model, tokenizer, sentences):
    max_length = 40
    sentenceList = []
    sentenceTokenList = []
    nerTokenList = []
    sentLengthList = []
    bertSentenceIDs = []
    bertMasks = []
    bertSequenceIDs = []
    sentence = ''
    wordIndexTokenPosList = []

    for sentence in sentences:
        # always start with [CLS] tokens
        sentenceTokens = ['[CLS]']
        nerTokens = ['[nerCLS]']
        # split sentence
        words = re.findall(r'\b\w+\b|\[.*?\]\{.*?\}\<.*?\>|\S', sentence)
        addDict = dict()
        wordIndexTokenPos = dict()
        for i, word in enumerate(words):
            tokens = tokenizer.tokenize(word)
            tokenLength = len(tokens)
            wordIndexTokenPos[i] = len(sentenceTokens)
            sentenceTokens += tokens
            # addDict['tokenLength'] = tokenLength
            nerTokens = ['nerB'] + ['nerX'] * (tokenLength - 1)
        wordIndexTokenPosList.append(wordIndexTokenPos)
        sentenceLength = min(max_length -1, len(sentenceTokens))
        sentLengthList.append(sentenceLength)

        # Create space for at least a final '[SEP]' token
        if sentenceLength >= max_length - 1:
            sentenceTokens = sentenceTokens[:max_length - 2]
            nerTokens = nerTokens[:max_length - 2]

        # add a ['SEP'] token and padding
        sentenceTokens += ['[SEP]'] + ['[PAD]'] * (max_length -1 - len(sentenceTokens))
        nerTokens += ['[nerSEP]'] + ['[nerPAD]'] * (max_length - 1 - len(nerTokens) )
        sentenceList.append(words)
        sentenceTokenList.append(sentenceTokens)

        bertSentenceIDs.append(tokenizer.convert_tokens_to_ids(sentenceTokens))
        bertMasks.append([1] * (sentenceLength + 1) + [0] * (max_length -1 - sentenceLength ))
        bertSequenceIDs.append([0] * (max_length))

        nerTokenList.append(nerTokens)

    bertSentenceIDs = np.array(bertSentenceIDs)
    bertMasks = np.array(bertMasks)
    bertSequenceIDs = np.array(bertSequenceIDs)

    preds = model.predict([bertSentenceIDs, bertMasks, bertSequenceIDs], batch_size=16)
    preds = np.array(np.argmax(preds, axis=2))

    return _word_to_class(sentenceList, preds, wordIndexTokenPosList)