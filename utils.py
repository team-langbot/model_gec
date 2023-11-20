import argparse
import importlib
import datetime

class DatasetFiles:
    def __init__(self, train_csv, test_csv, dev_csv, train_ner, test_ner, dev_ner):
        self.GEC_TRAIN_CSV = train_csv
        self.GEC_TEST_CSV = test_csv
        self.GEC_DEV_CSV = dev_csv
        self.GEC_TRAIN_NER = train_ner
        self.GEC_TEST_NER = test_ner
        self.GEC_DEV_NER = dev_ner

class Config:
    def __init__(self):
        self.USE_SMALL_DATASET = False
        self.ESSAY_COL = 'essay'
        self.CORRECTED_COL = 'corrected'
        self.INPUT_COL = 'input_text'
        self.OUTPUT_COL = 'target_text'
        self.RAW_DATA_FOLDER = './raw_data/csv'
        self.PROCESSED_DATA_FOLDER = './processed_data'
        self.STANFORD_PARSERS_FOLDER = './stanford-parser-full-2020-11-17'
        self.STANFORD_CORENLP_FOLDER = './stanford-corenlp-4.5.5'
        self.EXPLAINABLE_GEC_DATA_FOLDER = './Explainable_GEC/data/json'

        self.full_dataset = DatasetFiles('bert_train.csv', 'bert_test.csv', 'bert_dev.csv', 'bert_train.pkl', 'bert_test.pkl', 'bert_dev.pkl')

        self.small_dataset = DatasetFiles('bert_train_small.csv', 'bert_test_small.csv', 'bert_dev_small.csv', 'bert_train_small.pkl', 'bert_test_small.pkl', 'bert_dev_small.pkl')

        self.tiny_dataset = DatasetFiles('bert_train_tiny.csv', 'bert_test_tiny.csv', 'bert_dev_tiny.csv', 'bert_train_tiny.pkl', 'bert_test_tiny.pkl', 'bert_dev_tiny.pkl')

        self.two_classed_dataset = DatasetFiles('bert_train_two_classed.csv', 'bert_test_two_classed.csv', 'bert_dev_two_classed.csv', 'bert_train_two_classed.pkl', 'bert_test_two_classed.pkl', 'bert_dev_two_classed.pkl')

        # # Config files to train and evaluate Labeling-based **Error+Correction** model
        # self.EC_TRAIN_CONFIG = 'cfgs/train_error_correction.py'
        # self.EC_EVAL_CONFIG = 'cfgs/eval_error_correction.py'

        # # Config files to train and evaluate Labeling-based **Error+Correction+CE** model
        # self.ECC_TRAIN_CONFIG = 'cfgs/train_error_correction_ce.py'
        # self.ECC_EVAL_CONFIG = 'cfgs/eval_error_correction_ce.py'

        # # Config files to train and evaluate Labeling-based **Error+Correction+CE+Syntax** model
        # self.ECC_TRAIN_CONFIG = 'cfgs/train_error_correction_ce_syntax.py'
        # self.ECC_EVAL_CONFIG = 'cfgs/eval_error_correction_ce_syntax.py'

        self.CLASS_IDS = {
            'article': 'a',
            'gender agreement': 'ga',
            'gender and number agreement': 'gna',
            'number agreement': 'na',
        }

    def training_dataset(self):
        if self.USE_SMALL_DATASET:
            return self.small_dataset
        else:
            return self.full_dataset

class Training_config:
    def __init__(self, config_file):
        self.use_cuda = False

        # type=list, default=None, help="if is only inference and inference which data. Set None if is training", choices=[None, 'train', 'test', 'dev'])
        self.only_inference = None

        # type=str, default='bert', help="if use a pretrained_model")
        self.model = 'auto'

        # type=str, default='interactive', help="if with syntax", choices=['interactive', 'noninteractive'])
        self.model_type = 'interactive'

        # type=str, default="bert-large-cased", help="if use a pretrained_model")
        self.model_name = "bert-large-cased"
        self.model_name = "bert-base-multilingual-cased" # mBert
        self.model_name = "dccuchile/bert-base-spanish-wwm-uncased" #Original Beto

        #  type=str, default='', help="path to save the evaluation results")
        self.eval_res_dir = ''

        # type=bool, default=False, help="")
        self.only_eval = False

        # type=bool, default=False, help="if split dev to train set")
        self.split_dev = False

        # type=int, default=0, help="if reduce trainset to half")
        self.half_train = 0

        # type=int, default=0, help="if reduce evalset to half")
        self.half_eval = 0

        # type=str, default='test', help="name of the exp")
        self.exp_name = 'test'

        # type=str, default=None, help="name of the exp")
        self.output_dir = "outputs"

        # type=float, default=1e-5, help="learning rate")
        self.lr = 1e-5

        # type=int, default=10, help="")
        self.epochs = 10

        # type=int, default=8, help="")
        self.train_batch_size = 8

        # type=int, default=64, help="")
        self.eval_batch_size = 64

        # type=list, default=[], help="")
        self.train_dev_data = []

        # type=bool, default=False, help="")
        self.multi_loss = False

        # type=list, default=[0.5, 0.5], help="")
        self.loss_weight = [0.5, 0.5]

        # type=bool, default=False, help="")
        self.wo_token_labels = False

        # type=bool, default=False, help="")
        self.debug = False

        # type=list, default=[], help="")
        self.labels_list = []

        # type=bool, default=False, help="")
        self.with_errant = False

        # type=int, default=3, help="if add extra correction position embedding for indicating the position of the correction")
        self.max_correction_embeddings = 3

        # type=bool, default=False, help="")
        self.interactive_mode = False

        # type=list, default=[], help="")
        self.rule_data = []

        # type=list, default=[], help="")
        self.new_data = []

        # type=bool, default=False, help="parallel data, tgt+[SEP]+src")
        self.parallel = False

        # type=int, default=256, help="max_seq_length of BERT")
        self.max_seq_length = 256

        # type=str, default='', help="pkl file for test")
        self.test_file = ''

        # type=int, default=1, help="gpu number for distributed training")
        self.n_gpu = 1

        # type=int, default=None, help="Ensembled reference")
        self.ensemble_reference = None

        # type=bool, default=False, help="")
        self.use_multiprocessing = False

        # type=str, default='AdamW')
        self.optimizer = 'AdamW'

        # type=bool, default=False)
        self.parsing_embedding = False

        # type=bool, default=False)
        self.eval_all_checkpoint = False

        # type=bool, default=True)
        self.evaluate_each_epoch = True

        # type=bool, default=True)
        self.evaluate_during_training = True

        # type=bool, default=False)
        self.stbert_ensemble = False

        # type=bool, default=False)
        self.is_qk = False

        # type=bool, default=False)
        self.is_dense = False

        # type=int, default=2)
        self.mn = 2

        # type=int, default=0)
        self.two_linears = 0

        # type=int, default=1024)
        self.linear_hidden_size = 1024

        # type=bool, default=False)
        self.parsing_heads = False

        # type=str, default=None,)
        self.parsing_heads_reshape = None

        # type=bool, default=False)
        self.parsing_embedding_matrix = False

        # type=bool, default=False)
        self.parsing_embedding_for_embedding = False

        # type=bool, default=False)
        self.parsing_embedding_for_model_matrix_embedding = False

        # type=bool, default=False)
        self.only_first_parsing_order = False

        config_path = config_file.replace('/', '.')[:-3]

        config = importlib.import_module(config_path).config
        # time_str = datetime.now().strftime('%Y%m%d-%H%M%S')

        # update based on the config
        for k, v in config.items():
            if k not in ['custom']:
                setattr(self, k, v)

        self.BIO_labels = [
            "B-a",  "I-a",
            "B-ga",  "I-ga",
            "B-gna",  "I-gna",
            "B-na",  "I-na",
            "O"]
