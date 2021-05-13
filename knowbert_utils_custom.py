from typing import Union, List

from allennlp.common import Params
from allennlp.data import Instance, DataIterator, Vocabulary
from allennlp.common.file_utils import cached_path


from kb.include_all import TokenizerAndCandidateGenerator
from kb.bert_pretraining_reader import replace_candidates_with_mask_entity
from kb.knowbert_utils import KnowBertBatchifier, _extract_config_from_archive, _find_key

from knowbert_custom import *
from kb.wiki_linking_reader import LinkingReader


class KnowBertBatchifierCustom(KnowBertBatchifier):
    def __init__(self, 
                 model_archive, batch_size=32,
                 masking_strategy=None,
                 wordnet_entity_file=None, 
                 vocab_dir=None,
                 candidate_generator_params=None):

        # get bert_tokenizer_and_candidate_generator
        config = _extract_config_from_archive(cached_path(model_archive))

        if candidate_generator_params is None:
            # look for the bert_tokenizers and candidate_generator
            candidate_generator_params = _find_key(
                config['dataset_reader'].as_dict(), 'tokenizer_and_candidate_generator'
            )
            
            assert candidate_generator_params is not None, "There are no candidate_generator_params, use fully trained model or provide them via a parameter"

            if wordnet_entity_file is not None:
                candidate_generator_params['entity_candidate_generators']['wordnet']['entity_file'] = wordnet_entity_file

        self.tokenizer_and_candidate_generator = TokenizerAndCandidateGenerator.\
                from_params(Params(candidate_generator_params))
        self.tokenizer_and_candidate_generator.whitespace_tokenize = False

        assert masking_strategy is None or masking_strategy == 'full_mask'
        self.masking_strategy = masking_strategy

        # need bert_tokenizer_and_candidate_generator
        if vocab_dir is not None:
            vocab_params = Params({"directory_path": vocab_dir})
        else:
            vocab_params = config['vocabulary']
        self.vocab = Vocabulary.from_params(vocab_params)

        self.iterator = DataIterator.from_params(
            Params({"type": "basic", "batch_size": batch_size})
        )
        self.iterator.index_with(self.vocab)
    