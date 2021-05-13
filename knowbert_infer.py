import torch
from allennlp.models.archival import load_archive
from allennlp.nn.util import move_to_device
from allennlp.common.file_utils import cached_path
import fire
from allennlp.common import Params

#from kb.knowbert_utils import KnowBertBatchifier
from knowbert_utils_custom import KnowBertBatchifierCustom


def predict(sentences, model, batcher):
    batch = next(batcher.iter_batches(sentences, verbose=False))
    b = move_to_device(batch, 0)
    model_output = model(**b)
#     for batch in batcher.iter_batches(sentences, verbose=False):
#         b = move_to_device(batch, 0)
#         model_output = model(**b)

    linking_scores = model_output['wiki']['linking_scores']
    candidate_entities = batch['candidates']['wiki']['candidate_entities']['ids']
    candidate_spans = batch['candidates']['wiki']['candidate_spans']

    linker = model.soldered_kgs['wiki'].entity_linker
    predictions = linker._decode(linking_scores, candidate_spans, candidate_entities)
    return predictions, batch, linking_scores


def idx_to_name(model, idx):
    name = model.soldered_kgs['wiki'].entity_linker.vocab.get_token_from_index(idx, 'entity')
    return name


def idxs_to_tokens(batch_token_idxs, tokenizer):
    batch_sents = []
    for sent_idxs in batch_token_idxs:
        idxs = [idx for idx in sent_idxs.numpy() if idx > 0]
        tokens = tokenizer.convert_ids_to_tokens(idxs)
        batch_sents.append(tokens)
    return batch_sents


DEFAULT_MODEL_URL = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_model.tar.gz'


WIKI_CANDIDATE_GENERATOR = Params({
                        "type": "bert_tokenizer_and_candidate_generator",
                        "entity_candidate_generators": {
                            "wiki": {"type": "wiki"},
                        },
                        "entity_indexers":  {
                            "wiki": {
                                   "type": "characters_tokenizer",
                                   "tokenizer": {
                                       "type": "word",
                                       "word_splitter": {"type": "just_spaces"},
                                   },
                                   "namespace": "entity"
                                }
                        },
                        "bert_model_type": "bert-base-uncased",
                        "do_lower_case": True})


def main(model_url=DEFAULT_MODEL_URL):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model_archive_file = cached_path(model_url)

    archive = load_archive(model_archive_file)
    batcher = KnowBertBatchifierCustom(model_archive_file, candidate_generator_params=WIKI_CANDIDATE_GENERATOR)
    tokenizer = batcher.tokenizer_and_candidate_generator.bert_tokenizer

    model = archive.model
    model.to(device)
    model.eval()
    
    while True:
        text = input('Enter a sentence: ')
        sentences = [text]
        preds, batch, _ = predict(sentences, model, batcher)

        if len(preds) == 0:
            print('No entities found.')
            continue

        preds_with_names = []
        for in_batch_idx, span, entity_idx in preds:
            preds_with_names.append((in_batch_idx, span, idx_to_name(model, entity_idx),))

        token_idxs = batch['tokens']['tokens']
        batch_sents = idxs_to_tokens(token_idxs, tokenizer)

        for in_batch_idx, span, ent_name in preds_with_names:
            sent = batch_sents[in_batch_idx][:]
            start, end = span
            sent[start] = '(' + sent[start]
            sent[end] = sent[end] + f')[https://en.wikipedia.org/wiki/{ent_name}]'

            # Drop CLS and SEP
            sent = sent[1:-1]
            s = ''
            for tok in sent:
                s += tok[2:] if tok.startswith('##') else ' ' + tok

            print(s)


if __name__ == "__main__":
    fire.Fire(main)
