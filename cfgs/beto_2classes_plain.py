config = {
    "exp_name" : '/beto/plain',
    "eval_res_dir": '',
    "only_inference" : None,
    "train_dev_data": ['processed_data/bert_train_two_classed_plain.csv',
                       'processed_data/bert_dev_two_classed_plain.csv',
                       'processed_data/bert_test_two_classed_plain.csv',],
    "max_correction_embeddings": 3, # set 0 if no correction embeddings, else 3
    "lr": 1e-5,
    "max_seq_length": 256,
    "parsing_embedding": True,
    "parsing_embedding_for_embedding": True,
    "model_type": 'non_interactive',
    "model": 'bert',
    "model_name": "dccuchile/bert-base-spanish-wwm-uncased", #Original Beto
    "exp_name": "beto_cows_l2h_two_classes",
    "labels_list": [
            "B-ga",
            "I-ga",
            "B-na",
            "I-na",
            "O"
        ],
    "epochs": 5
}
