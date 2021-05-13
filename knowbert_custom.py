import torch
from torch.nn import PairwiseDistance, CosineSimilarity, CosineEmbeddingLoss

import math
from typing import Dict, List
from overrides import overrides
import random
import copy

from allennlp.data import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.regularizers import RegularizerApplicator

from kb.knowbert import EntityLinkingWithCandidateMentions, EntityDisambiguator, DotAttentionWithPrior, SolderedKG
from kb.common import EntityEmbedder
from kb.include_all import *
from kb.common import get_dtype_for_module, set_requires_grad, \
    extend_attention_mask_for_bert, init_bert_weights, F1Metric


class DotAttentionWithPriorCustom(DotAttentionWithPrior):
    def __init__(self, 
                 output_feed_forward_hidden_dim: int = 100,
                 weighted_entity_threshold: float = None,
                 null_embedding: torch.Tensor = None,
                 initializer_range: float = 0.02,
                 predict_embeddings: bool = False,
                 similarity: str = 'dotprod'):
        super().__init__(output_feed_forward_hidden_dim,
                         weighted_entity_threshold,
                         null_embedding,
                         initializer_range)
        self._predict_embeddings = predict_embeddings
        self._similarity = similarity
        print('Predict embeddings:', predict_embeddings)
    
    @overrides
    def forward(self,
            projected_span_representations,
            candidate_entity_embeddings,
            candidate_entity_prior,
            entity_mask):
        
        # dot product between span embedding and entity embeddings, scaled
        # by sqrt(dimension) as in Transformer
        # (batch_size, num_spans, num_candidates)
        if self._similarity == 'cosine':
            scores = CosineSimilarity(dim=-1)(projected_span_representations.unsqueeze(-2), 
                                              candidate_entity_embeddings)
        elif self._similarity == 'dotprod':
            scores = torch.sum(
                projected_span_representations.unsqueeze(-2) * candidate_entity_embeddings,
                dim=-1
            ) / math.sqrt(candidate_entity_embeddings.shape[-1])
        elif self._similarity == 'euclead':
            euclead_sim = lambda x,y: torch.sqrt(torch.pow(x.unsqueeze(-2) - y, 2).sum(-1)) # we actually do not need sqrt
            scores = 1. / (1. + euclead_sim(projected_span_representations, candidate_entity_embeddings))
        else:
            raise ValueError(f'Wrong similarity metric: {self._similarity}')

        # compute the final score
        # the prior needs to be input as float32 due to half not supported on
        # cpu.  so need to cast it here.
        if self._predict_embeddings:
            linking_score = scores.squeeze(-1)
        else:
            dtype = list(self.parameters())[0].dtype
            scores_with_prior = torch.cat(
                [scores.unsqueeze(-1), candidate_entity_prior.unsqueeze(-1).to(dtype)],
                dim=-1
            )

            # (batch_size, num_spans, num_candidates)
            linking_score = self.out_layer_2(
                torch.nn.functional.relu(self.out_layer_1(scores_with_prior))
            ).squeeze(-1)

        #print('LS shpae:', linking_score.shape)

        # mask out the invalid candidates
        invalid_candidate_mask = ~entity_mask

        linking_scores = linking_score.masked_fill(invalid_candidate_mask, -10000.0)
        return_dict = {'linking_scores': linking_scores}

        weighted_entity_embeddings = self._get_weighted_entity_embeddings(
                linking_scores, candidate_entity_embeddings
        )
        return_dict['weighted_entity_embeddings'] = weighted_entity_embeddings

        return return_dict


class EntityDisambiguatorCustom(EntityDisambiguator):
    def __init__(self, 
                 contextual_embedding_dim,
                 entity_embedding_dim: int,
                 entity_embeddings: torch.nn.Embedding,
                 max_sequence_length: int = 512,
                 span_encoder_config: Dict[str, int] = None,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02,
                 weighted_entity_threshold: float = None,
                 null_entity_id: int = None,
                 include_null_embedding_in_dot_attention: bool = False,
                 predict_embeddings: bool = False,
                 similarity: str = 'dotprod',
                 entity_embedding_preprocessing : bool = True,
                 span_tanh : bool = False):
        
        super().__init__(contextual_embedding_dim,
                         entity_embedding_dim,
                         entity_embeddings,
                         max_sequence_length,
                         span_encoder_config,
                         dropout,
                         output_feed_forward_hidden_dim,
                         initializer_range,
                         weighted_entity_threshold,
                         null_entity_id,
                         include_null_embedding_in_dot_attention)
        
        if weighted_entity_threshold is not None or include_null_embedding_in_dot_attention:
            if hasattr(self.entity_embeddings, 'get_null_embedding'):
                null_embedding = self.entity_embeddings.get_null_embedding()
            else:
                null_embedding = self.entity_embeddings.weight[null_entity_id, :]
        else:
            null_embedding = None
        if predict_embeddings:
            self.dot_attention_with_prior = DotAttentionWithPriorCustom(
                     output_feed_forward_hidden_dim,
                     weighted_entity_threshold,
                     null_embedding,
                     initializer_range,
                     predict_embeddings,
                     similarity,
            )
        
        self._predict_embeddings = predict_embeddings
        self._entity_embedding_preprocessing = entity_embedding_preprocessing
        self._span_tanh = span_tanh
        
        if self._span_tanh:
            self._tanh_layer = torch.nn.Linear(300, 300)
            self._tanh_activation = torch.nn.Tanh()
    
    @overrides
    def forward(self,
                contextual_embeddings: torch.Tensor,
                mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs
        ):
        """
        contextual_embeddings = (batch_size, timesteps, dim) output
            from language model
        mask = (batch_size, num_times)
        candidate_spans = (batch_size, max_num_spans, 2) with candidate
            mention spans. This gives the start / end location for each
            span such span i in row k has:
                start, end = candidate_spans[k, i, :]
                span_embeddings = contextual_embeddings[k, start:end, :]
            it is padded with -1
        candidate_entities = (batch_size, max_num_spans, max_entity_ids)
            padded with 0
        candidate_entity_prior = (batch_size, max_num_spans, max_entity_ids)
            with prior probability of each candidate entity.
            0 <= candidate_entity_prior <= 1 and candidate_entity_prior.sum(dim=-1) == 1
        Returns:
            linking sccore to each entity in each span
                (batch_size, max_num_spans, max_entity_ids)
            masked with -10000 for invalid links
        """
        # get the candidate entity embeddings
        # (batch_size, num_spans, num_candidates, entity_embedding_dim)
        candidate_entity_embeddings = self.entity_embeddings(candidate_entities)
        if self._entity_embedding_preprocessing: 
            candidate_entity_embeddings = self.kg_layer_norm(candidate_entity_embeddings.contiguous())

        # project to entity embedding dim
        # (batch_size, timesteps, entity_dim)
        projected_bert_representations = self.bert_to_kg_projector(contextual_embeddings)

        # compute span representations
        span_mask = (candidate_spans[:, :, 0] > -1)#.long() # need bool  ?
        # (batch_size, num_spans, embedding_dim)
        projected_span_representations = self.span_extractor(
            projected_bert_representations,
            candidate_spans,
            mask,
            span_mask
        )
        projected_span_representations = self.projected_span_layer_norm(projected_span_representations.contiguous())

        # run the span transformer encoders
        if self.span_encoder is not None:
            projected_span_representations = self._run_span_encoders(
                projected_span_representations, span_mask
            )[-1]
            
        if self._span_tanh:
            projected_span_representations = self._tanh_activation(self._tanh_layer(projected_span_representations))

        entity_mask = candidate_entities > 0
        return_dict = self.dot_attention_with_prior(
                    projected_span_representations,
                    candidate_entity_embeddings,
                    candidate_entity_priors,
                    entity_mask)

        return_dict['projected_span_representations'] = projected_span_representations
        return_dict['projected_bert_representations'] = projected_bert_representations

        return return_dict

            
@Model.register("entity_linking_with_candidate_mentions_custom")
class EntityLinkingWithCandidateMentionsCustom(EntityLinkingWithCandidateMentions):
    def __init__(self, 
                 vocab: Vocabulary,
                 kg_model: Model = None,
                 entity_embedding: Embedding = None,
                 concat_entity_embedder: EntityEmbedder = None,
                 contextual_embedding_dim: int = None,
                 span_encoder_config: Dict[str, int] = None,
                 margin: float = 0.2,
                 decode_threshold: float = 0.0,
                 loss_type: str = 'margin',
                 max_sequence_length: int = 512,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02,
                 include_null_embedding_in_dot_attention: bool = False,
                 namespace: str = 'entity',
                 regularizer: RegularizerApplicator = None,
                 predict_embeddings: bool = False,
                 entity_embedding_preprocessing: bool = True,
                 similarity: str ='dotprod', 
                 random_candidates: float = 0.,
                 hard_negative_mining: bool = False,
                 eval_no_candidates: bool = False,
                 no_index_candidates: bool = False,
                 entity_encoder: bool = False,
                 span_tanh: bool = False,
                 device_for_dotprod = None):
        
        print('predict_embeddings', predict_embeddings)
        print('entity_embedding_preprocessing', entity_embedding_preprocessing)
        print('similarity', similarity)
        
        self._random_candidates = (int(random_candidates) 
                                   if random_candidates.is_integer() 
                                   else int(random_candidates * entity_embedding.weight.shape[0]))
        self._hard_negative_mining = hard_negative_mining
        self._margin = margin
        self.real_loss_type = loss_type
        if loss_type == 'triplet' or loss_type == 'cosine' or loss_type == 'cosine_custom':
            loss_type = 'margin'
        
        super().__init__(vocab,
                 kg_model,
                 entity_embedding,
                 concat_entity_embedder,
                 contextual_embedding_dim,
                 span_encoder_config,
                 margin,
                 decode_threshold,
                 loss_type,
                 max_sequence_length,
                 dropout,
                 output_feed_forward_hidden_dim,
                 initializer_range,
                 include_null_embedding_in_dot_attention,
                 namespace,
                 regularizer)

        self._dotprod_embedding = None
        if predict_embeddings:
            self._device_for_dotprod = torch.device(device_for_dotprod) if device_for_dotprod is not None else None
                        
            self._entity_embedding = entity_embedding
            
            num_embeddings_passed = sum(
            [kg_model is not None, entity_embedding is not None, concat_entity_embedder is not None]
            )
            if num_embeddings_passed != 1:
                raise ValueError("Linking model needs either a kg factorisation model or an entity embedding.")

            elif kg_model is not None:
                entity_embedding = kg_model.get_entity_embedding()
                entity_embedding_dim  = entity_embedding.embedding_dim

            elif entity_embedding is not None:
                entity_embedding_dim  = entity_embedding.get_output_dim()

            elif concat_entity_embedder is not None:
                entity_embedding_dim  = concat_entity_embedder.get_output_dim()
                set_requires_grad(concat_entity_embedder, False)
                entity_embedding = concat_entity_embedder

            if loss_type == 'margin':
                weighted_entity_threshold = decode_threshold
            else:
                weighted_entity_threshold = None

            null_entity_id = self.vocab.get_token_index('@@NULL@@', namespace)
            assert null_entity_id != self.vocab.get_token_index('@@UNKNOWN@@', namespace)
            
            self.disambiguator = EntityDisambiguatorCustom(
                     contextual_embedding_dim,
                     entity_embedding_dim=entity_embedding_dim,
                     entity_embeddings=entity_embedding,
                     max_sequence_length=max_sequence_length,
                     span_encoder_config=span_encoder_config,
                     dropout=dropout,
                     output_feed_forward_hidden_dim=output_feed_forward_hidden_dim,
                     initializer_range=initializer_range,
                     weighted_entity_threshold=weighted_entity_threshold,
                     include_null_embedding_in_dot_attention=include_null_embedding_in_dot_attention,
                     null_entity_id=null_entity_id,
                     predict_embeddings=predict_embeddings,
                     entity_embedding_preprocessing=entity_embedding_preprocessing,
                     similarity=similarity,
                     span_tanh=span_tanh)
            
            if self.real_loss_type == 'cosine':
                self.loss = CosineEmbeddingLoss(margin=margin)
            
            self._eval_no_candidates = eval_no_candidates
            
            if self._eval_no_candidates:
                self._entity_indexes = torch.tensor([e for e in range(self._entity_embedding.weight.shape[0])])
                #self._emb_norms = torch.sqrt(torch.pow(self._entity_embedding.weight, 2).sum(-1, keepdim=True))
                
            self._no_index_candidates = no_index_candidates
            
            self._entity_encoder = entity_encoder
            if self._entity_encoder:
                self._entity_encoder_layer = torch.nn.Linear(300, 300)
                self._tanh_activation = torch.nn.Tanh()
                
            self._entity_embedding_preprocessing = entity_embedding_preprocessing
    
    @overrides
    def _compute_loss(self,
                      candidate_entities,
                      candidate_spans,
                      linking_scores,
                      gold_entities,
                      projected_span_representations):
        if self.real_loss_type == 'margin':
            return self._compute_margin_loss(
                    candidate_entities, candidate_spans, linking_scores, gold_entities)
        elif self.real_loss_type == 'softmax':
            return self._compute_softmax_loss(
                    candidate_entities, candidate_spans, linking_scores, gold_entities)
        elif self.real_loss_type == 'triplet':
            return self._triplet_loss(candidate_entities, candidate_spans, 
                                      linking_scores, gold_entities, 
                                      projected_span_representations)
        elif self.real_loss_type == 'cosine':
            return self._cosine_loss(candidate_entities, candidate_spans, 
                                      linking_scores, gold_entities, 
                                      projected_span_representations)
        elif self.real_loss_type == 'cosine_custom':
            return self._cosine_custom_loss(candidate_entities, candidate_spans, 
                                            linking_scores, gold_entities)
        else:
            raise ValueError(f'Wrong loss type: {self.real_loss_type}')
            
            
    def _cosine_custom_loss(self, candidate_entities, 
                            candidate_spans, 
                            linking_scores, gold_entities):
        # compute loss
        # in End-to-End Neural Entity Linking
        # loss = max(0, gamma - score) if gold mention
        # loss = max(0, score) if not gold mention
        #
        # torch.nn.MaxMarginLoss(x1, x2, y) = max(0, -y * (x1 - x2) + gamma)
        #   = max(0, -x1 + x2 + gamma)  y = +1
        #   = max(0, gamma - x1) if x2 == 0, y=+1
        #
        #   = max(0, x1 - gamma) if y==-1, x2=0

        candidate_mask = candidate_entities > 0
        # (num_entities, )
        non_masked_scores = linking_scores[candidate_mask]

        # broadcast gold ids to all candidates
        num_candidates = candidate_mask.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
                    1, 1, num_candidates
        )
        # compute +1 / -1 labels for whether each candidate is gold
        positive_labels = (broadcast_gold_entities == candidate_entities)
        negative_labels = (broadcast_gold_entities != candidate_entities)
        #labels = (positive_labels - negative_labels).to(dtype=get_dtype_for_module(self))
        # finally select the non-masked candidates
        # (num_entities, ) with +1 / -1
        non_masked_positive_labels = positive_labels[candidate_mask]
        non_masked_negative_labels = negative_labels[candidate_mask]

        negative_scores = non_masked_scores[non_masked_negative_labels]
        zeros_negative = torch.zeros_like(negative_scores)
        loss = ((1. - non_masked_scores[non_masked_positive_labels]).sum() 
                + torch.max(zeros_negative, non_masked_scores[non_masked_negative_labels] - self._margin).sum())

        # metrics
        self._compute_f1(linking_scores, candidate_spans,
                         candidate_entities,
                         gold_entities)

        return {'loss': loss}
        
    def _triplet_loss(self, candidate_entities, candidate_spans, 
                      linking_scores, gold_entities, 
                      projected_span_representations):
        # not exactly triplet loss, but with euclead norm of embeddings
        
        # compute loss
        # in End-to-End Neural Entity Linking
        # loss = max(0, gamma - score) if gold mention
        # loss = max(0, score) if not gold mention
        #
        # torch.nn.MaxMarginLoss(x1, x2, y) = max(0, -y * (x1 - x2) + gamma)
        #   = max(0, -x1 + x2 + gamma)  y = +1
        #   = max(0, gamma - x1) if x2 == 0, y=+1
        #
        #   = max(0, x1 - gamma) if y==-1, x2=0

        candidate_mask = candidate_entities > 0
        # (num_entities, )
        #non_masked_scores = linking_scores[candidate_mask]

        # broadcast gold ids to all candidates
        num_candidates = candidate_mask.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
                    1, 1, num_candidates
        )
        # compute +1 / -1 labels for whether each candidate is gold
        positive_labels = (broadcast_gold_entities == candidate_entities).long()
        negative_labels = (broadcast_gold_entities != candidate_entities).long()
        labels = (positive_labels - negative_labels).to(dtype=get_dtype_for_module(self))
        # finally select the non-masked candidates
        # (num_entities, ) with +1 / -1
        
        entity_embeddings = self.disambiguator.kg_layer_norm(self._entity_embedding(candidate_entities))
        entity_embeddings = entity_embeddings[candidate_mask]
        non_masked_labels = labels[candidate_mask]
        
        span_embeds = projected_span_representations.unsqueeze(-2).repeat(1, 1, candidate_mask.shape[-1], 1)
        span_embeds = span_embeds[candidate_mask]

        scores = torch.pow(span_embeds - entity_embeddings, 2).sum(-1)
        non_masked_labels = -non_masked_labels
        
        loss = self.loss(
                scores, torch.zeros_like(non_masked_labels),
                non_masked_labels
        )

        # metrics
        self._compute_f1(linking_scores, candidate_spans,
                         candidate_entities,
                         gold_entities)

        return {'loss': loss}
    
    def _sample_with_hard_negative_mining(self, span_repr, gold_entities):
        with torch.no_grad():
            emb_indices = random.sample(list(range(self._entity_embedding.weight.shape[0])), 25000)
            
            result = []
            sims = torch.matmul(span_repr, self._entity_embedding.weight[emb_indices].T)
            for b in range(span_repr.shape[0]):
                for j in range(span_repr.shape[1]):
                    sim_ind = torch.topk(sims[b, j], self._random_candidates)[1]
                    sim_ind_filtered = []
                    for e in sim_ind:
                        if e == gold_entities[b, j]:
                            e = random.randint(0, self._entity_embedding.weight.shape[0] - 1)
                            
                        sim_ind_filtered.append(e)

                    result.append(torch.LongTensor(sim_ind_filtered))

            candidates = torch.cat(result).reshape(span_repr.shape[:-1] + (-1,))
            return candidates
    
    def _cosine_loss(self, candidate_entities, candidate_spans, 
                     linking_scores, gold_entities, 
                     projected_span_representations):
        # not exactly triplet loss, but with euclead norm of embeddings
        
        # compute loss
        # in End-to-End Neural Entity Linking
        # loss = max(0, gamma - score) if gold mention
        # loss = max(0, score) if not gold mention
        #
        # torch.nn.MaxMarginLoss(x1, x2, y) = max(0, -y * (x1 - x2) + gamma)
        #   = max(0, -x1 + x2 + gamma)  y = +1
        #   = max(0, gamma - x1) if x2 == 0, y=+1
        #
        #   = max(0, x1 - gamma) if y==-1, x2=0

        orig_entities = candidate_entities
        if self._random_candidates > 0:
            if self._hard_negative_mining:
                random_candidates_tensor = self._sample_with_hard_negative_mining(projected_span_representations, gold_entities).cuda()
            else:
                n_sample = self._random_candidates * candidate_entities.shape[0] * candidate_entities.shape[1]
                random_candidates = random.sample(list(range(self._entity_embedding.weight.shape[0])), n_sample)
                random_candidates_tensor = torch.LongTensor(random_candidates).reshape(candidate_entities.shape[0], 
                                                                                       candidate_entities.shape[1], -1).cuda()
                
            if self._no_index_candidates:
#                 broadcast_gold_entities = gold_entities.repeat(
#                     1, 1, candidate_entities.shape[-1]
#                 )
#                 # compute +1 / -1 labels for whether each candidate is gold
#                 positive_labels = (broadcast_gold_entities == candidate_entities)
                
#                 pos_cands = [] # TODO: refactor
#                 for i in range(candidate_entities.shape[0]):
#                     tmp_res = []
#                     for j in range(candidate_entities.shape[1]):
#                         with torch.no_grad():
#                             tps = candidate_entities[i, j][positive_labels[i,j]]
                        
#                         if tps.shape[0] == 0:
#                             tmp_res.append([0])
#                         else:
#                             tmp_res.append([tps[0].item()])
                        
#                     pos_cands.append(tmp_res)
                    
#                 pos_cands = torch.tensor(pos_cands).cuda()
                pos_cands = ((gold_entities != self.null_entity_id).long() * gold_entities)
                candidate_entities = torch.cat((pos_cands, random_candidates_tensor), dim=2)
            else:
                candidate_entities = torch.cat((candidate_entities, random_candidates_tensor), dim=2)
        
        candidate_mask = candidate_entities > 0
        # (num_entities, )
        #non_masked_scores = linking_scores[candidate_mask]

        # broadcast gold ids to all candidates
        num_candidates = candidate_mask.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
                    1, 1, num_candidates
        )
        # compute +1 / -1 labels for whether each candidate is gold
        positive_labels = (broadcast_gold_entities == candidate_entities).long()
        
        negative_labels = (broadcast_gold_entities != candidate_entities).long()
        labels = (positive_labels - negative_labels).to(dtype=get_dtype_for_module(self))
        # finally select the non-masked candidates
        # (num_entities, ) with +1 / -1
        
        # @NULL@
        entity_embeddings = self._entity_embedding(candidate_entities)
        if self._entity_encoder:
            entity_embeddings = self._tanh_activation(self._entity_encoder_layer(entity_embeddings))
            
        if self._entity_embedding_preprocessing:
            entity_embeddings = self.disambiguator.kg_layer_norm(entity_embeddings)
            
        entity_embeddings = entity_embeddings[candidate_mask]
        non_masked_labels = labels[candidate_mask]
        
        span_embeds = projected_span_representations.unsqueeze(-2).repeat(1, 1, candidate_mask.shape[-1], 1)
        span_embeds = span_embeds[candidate_mask]
        
        # CosineEmbeddingLoss
        loss = self.loss(entity_embeddings, span_embeds, non_masked_labels)

        if self._eval_no_candidates:
            with torch.no_grad():
                emb_ent = self._dotprod_embedding.weight
                if self._entity_encoder:
                    emb_ent = self._tanh_activation(self._entity_encoder_layer(emb_ent)) 
                    
                if self._entity_embedding_preprocessing:
                    emb_ent = copy.deepcopy(self.disambiguator.kg_layer_norm).to(self._device_for_dotprod)(emb_ent)
                    
                emb_ent = emb_ent / torch.norm(emb_ent, p=2, dim=-1, keepdim=True)
                
                emb_span = projected_span_representations / torch.norm(projected_span_representations, p=2, dim=-1, keepdim=True)
                
                if self._device_for_dotprod is not None:
                    linking_scores = torch.matmul(emb_span.to(self._device_for_dotprod), emb_ent.T)
                else:
                    linking_scores = torch.matmul(emb_span, emb_ent.T)
                
                orig_entities = self._entity_indexes.repeat(candidate_spans.shape[0], candidate_spans.shape[1], 1)
            
        # metrics
        self._compute_f1(linking_scores, 
                         candidate_spans,
                         orig_entities,
                         gold_entities)

        return {'loss': loss}
    
    @overrides
    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs):
        if self._dotprod_embedding is None:
            if self._device_for_dotprod is not None:
                self._dotprod_embedding = copy.deepcopy(self._entity_embedding).to(self._device_for_dotprod)
            else:
                self._dotprod_embedding = self._entity_embedding
        
        disambiguator_output = self.disambiguator(
            contextual_embeddings=contextual_embeddings,
            mask=tokens_mask,
            candidate_spans=candidate_spans,
            candidate_entities=candidate_entities['ids'],
            candidate_entity_priors=candidate_entity_priors,
            candidate_segment_ids=candidate_segment_ids,
            **kwargs
        )

        linking_scores = disambiguator_output['linking_scores']

        return_dict = disambiguator_output

        if 'gold_entities' in kwargs:
            loss_dict = self._compute_loss(
                    candidate_entities['ids'],
                    candidate_spans,
                    linking_scores,
                    kwargs['gold_entities']['ids'],
                    disambiguator_output['projected_span_representations']
                
            )
            return_dict.update(loss_dict)

        return return_dict
    

@Model.register("soldered_kg_custom")    
class SolderedKGCustom(SolderedKG):
    def __init__(self,  
                 vocab: Vocabulary,
                 entity_linker: Model,
                 span_attention_config: Dict[str, int],
                 should_init_kg_to_bert_inverse: bool = True,
                 freeze: bool = False,
                 regularizer: RegularizerApplicator = None):
        super().__init__(vocab,
                 entity_linker,
                 span_attention_config,
                 should_init_kg_to_bert_inverse,
                 freeze,
                 regularizer)
    
    @overrides
    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs):

        linker_output = self.entity_linker(
                contextual_embeddings, tokens_mask,
                candidate_spans, candidate_entities, 
                candidate_entity_priors,
                candidate_segment_ids, **kwargs)

        # update the span representations with the entity embeddings
        span_representations = linker_output['projected_span_representations']
        weighted_entity_embeddings = linker_output['weighted_entity_embeddings']
        spans_with_entities = self.weighted_entity_layer_norm(
                (span_representations +
                self.dropout(weighted_entity_embeddings)).contiguous()
        )

        # now run self attention between bert and spans_with_entities
        # to update bert.
        # this is done in projected dimension
        entity_mask = candidate_spans[:, :, 0] > -1
        span_attention_output = self.span_attention_layer(
                linker_output['projected_bert_representations'],
                spans_with_entities,
                entity_mask
        )
        projected_bert_representations_with_entities = span_attention_output['output']
        entity_attention_probs = span_attention_output["attention_probs"]

        # finally project back to full bert dimension!
        bert_representations_with_entities = self.kg_to_bert_projection(
                projected_bert_representations_with_entities
        )
        new_contextual_embeddings = self.output_layer_norm(
                (contextual_embeddings + self.dropout(bert_representations_with_entities)).contiguous()
        )

        return_dict = {'entity_attention_probs': entity_attention_probs,
                       'contextual_embeddings': new_contextual_embeddings,
                       'linking_scores': linker_output['linking_scores'],
                       'projected_span_representations': linker_output['projected_span_representations']}
        if 'loss' in linker_output:
            return_dict['loss'] = linker_output['loss']

        return return_dict
