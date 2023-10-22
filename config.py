class Config:
    def __init__(self):
        self.ESSAY_COL = 'essay'
        self.CORRECTED_COL = 'corrected'
        self.INPUT_COL = 'input_text'
        self.OUTPUT_COL = 'target_text'
        self.RAW_DATA_FOLDER = './raw_data/csv'
        self.PROCESSED_DATA_FOLDER = './processed_data'
        self.STANFORD_PARSERS_FOLDER = './stanford-parser-full-2020-11-17'
        self.STANFORD_CORENLP_FOLDER = './stanford-corenlp-4.5.5'
        self.EXPLAINABLE_GEC_DATA_FOLDER = './Explainable_GEC/data/json'
