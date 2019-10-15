{
    "dataset_reader": {
        "type": "bert_crf_tagger",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "/home/liangjj/Bert/chinese_L-12_H-768_A-12/",
                "do_lowercase": true,
                "use_starting_offsets": false
            }
        },
    },
  "train_data_path": "/home/liangjj/fin_ner/data/final_train.txt",
  "validation_data_path": "/home/liangjj/fin_ner/data/final_dev.txt",
    "model": {
        "type": "bert_crf_tagger",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets", "bert-type-ids"],
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "/home/liangjj/Bert/chinese_L-12_H-768_A-12/",
                    "requires_grad": true,
                    "top_layer_only": true
                }
            }
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "text",
                "num_tokens"
            ]
        ],
        "batch_size":6 ,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "num_epochs":20 ,
        "grad_norm": 5,
        "patience": 5,
        "validation_metric": "+f1-measure-overall",
        "cuda_device":3 ,
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-5,
            "warmup": 0.1,
            "t_total": 60000
        }
    }
}
