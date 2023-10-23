config = {
    "exp_name" : '/error_correction/ner_1e5_bs32_bert_large/checkpoint-n', # best checkpoint path
    "eval_res_dir" : 'eval/dev.pkl',
    "only_inference" : ['dev'], # choose from ['dev', 'test'] or both
    "train_dev_data": ['processed_data/bert_train.pkl',
                       'processed_data/bert_dev.pkl',
                       'processed_data/bert_test.pkl',],
    "max_correction_embeddings": 0, # set 0 if no correction embeddings, else 3
    "debug": False,
    "max_seq_length": 256,
    "model_type": 'non_interactive'
}
