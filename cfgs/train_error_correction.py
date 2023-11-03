config = {
    "exp_name" : '/error_correction/ner_1e5_bs32_bert_large',
    "eval_res_dir": '',
    "only_inference" : None,
    "train_dev_data": ['processed_data/bert_train_small.pkl',
                       'processed_data/bert_dev_small.pkl',
                       'processed_data/bert_test_small.pkl',],
    "max_correction_embeddings": 0, # set 0 if no correction embeddings, else 3
    "lr": 1e-5,
    "debug": False,
    "max_seq_length": 256,
    "model_type": 'non_interactive'
}
