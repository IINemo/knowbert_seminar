{
    "dataset_reader": {
        "type": "aida_wiki_linking",
        "entity_disambiguation_only": false,
        "entity_indexer": {
            "type": "characters_tokenizer",
            "namespace": "entity",
            "tokenizer": {
                "type": "word",
                "word_splitter": {
                    "type": "just_spaces"
                }
            }
        },
        "should_remap_span_indices": false,
        "token_indexers": {
            "tokens": {
                "type": "bert-pretrained",
                "do_lowercase": true,
                "max_pieces": 512,
                "pretrained_model": "bert-base-uncased",
                "use_starting_offsets": true
            }
        }
    },
    "iterator": {
        "iterator": {
            "type": "cross_sentence_linking",
            "batch_size": 32,
            "bert_model_type": "bert-base-uncased",
            "do_lower_case": true,
            "entity_indexer": {
                "type": "characters_tokenizer",
                "namespace": "entity",
                "tokenizer": {
                    "type": "word",
                    "word_splitter": {
                        "type": "just_spaces"
                    }
                }
            },
            "id_type": "wiki",
            "mask_candidate_strategy": "none",
            "max_predictions_per_seq": 0,
            "use_nsp_label": false
        },
        "type": "self_attn_bucket",
        "batch_size_schedule": "base-12gb-fp32"
    },
    "model": {
        "type": "knowbert",
        "bert_model_name": "bert-base-uncased",
        "mode": "entity_linking",
        "soldered_kgs": {
            "wiki": {
                "type": "soldered_kg_custom",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions_custom",
                    "contextual_embedding_dim": 768,
                    "decode_threshold": 0.27,
                    "entity_embedding": {
                        "embedding_dim": 300,
                        "pretrained_file": "s3://allennlp/knowbert/wiki_entity_linking/entities_glove_format.gz",
                        "sparse": false,
                        "trainable": false,
                        "vocab_namespace": "entity"
                    },
                    "predict_embeddings": true,
                    "entity_embedding_preprocessing": true,
                    "entity_encoder" : false,
                    "eval_no_candidates": true,
                    "hard_negative_mining": false,
                    "loss_type": "cosine",
                    "margin": 0.2,
                    "random_candidates": 0,
                    "similarity": "cosine",
                    "span_tanh" : true,
                    "no_index_candidates" : false,
                    "span_encoder_config": {
                        "hidden_size": 300,
                        "intermediate_size": 1024,
                        "num_attention_heads": 4,
                        "num_hidden_layers": 1
                    }
                },
                "span_attention_config": {
                    "hidden_size": 300,
                    "intermediate_size": 1024,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 1
                }
            }
        },
        "soldered_layers": {
            "wiki": 9
        }
    },
    "train_data_path": "s3://allennlp/knowbert/wiki_entity_linking/aida_train.txt",
    "validation_data_path": "s3://allennlp/knowbert/wiki_entity_linking/aida_dev.txt",
    "trainer": {
        "cuda_device": 0,
        "gradient_accumulation_batch_size": 32,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 10,
            "num_steps_per_epoch": 434
        },
        "num_epochs": 10,
        "num_serialized_models_to_keep": 2,
        "optimizer": {
            "type": "bert_adam",
            "lr": 0.001,
            "max_grad_norm": 1,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "t_total": -1,
            "weight_decay": 0.01
        },
        "should_log_learning_rate": true,
        "validation_metric": "+wiki_el_f1"
    },
    "vocabulary": {
        "directory_path": "s3://allennlp/knowbert/models/vocabulary_wiki.tar.gz"
    }
}