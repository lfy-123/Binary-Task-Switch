from transformers import RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaModel
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings, RobertaSelfAttention, RobertaSelfOutput, RobertaAttention,
    RobertaIntermediate, RobertaOutput, RobertaLayer, RobertaEncoder, RobertaPooler,
    RobertaPreTrainedModel, RobertaConfig, create_position_ids_from_input_ids
)

class KNNEnhancedRobertaEmbeddings(RobertaEmbeddings):
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config, binary_list_dict=None, length_list_dict=None):
        super().__init__(config)
        self.LayerNorm = KNNEnhancedLayerNorm(config.hidden_size, eps=config.layer_norm_eps, binary_list_dict=binary_list_dict, length_list_dict=length_list_dict)
        self.LayerNorm.flag = "embeddings"
        
        
        self.binary_list_dict = binary_list_dict
        self.length_list_dict = length_list_dict
        self.binary_weights = None
        self.word_embeddings_delta_weight = 0
        self.position_embeddings_delta_weight = 0
        self.token_type_embeddings_delta_weight = 0

    def set_delta_by_binary_weight(self):
        if self.binary_weights is None:
            self.word_embeddings_delta_weight = 0
            self.position_embeddings_delta_weight = 0
            self.token_type_embeddings_delta_weight = 0
            return

        device = self.word_embeddings.weight.device
        self.word_embeddings_delta_weight = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx]["word_embeddings.weight"].to(device) * self.binary_list_dict[idx]["word_embeddings.weight"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)
        self.position_embeddings_delta_weight = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx]["position_embeddings.weight"].to(device) * self.binary_list_dict[idx]["position_embeddings.weight"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)
        self.token_type_embeddings_delta_weight = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx]["token_type_embeddings.weight"].to(device) * self.binary_list_dict[idx]["token_type_embeddings.weight"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)

        self.LayerNorm.binary_weights = self.binary_weights
        self.LayerNorm.set_delta_by_binary_weight()


    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = F.embedding(
                input_ids,
                self.word_embeddings.weight + self.word_embeddings_delta_weight,
                self.word_embeddings.padding_idx,
                self.word_embeddings.max_norm,
                self.word_embeddings.norm_type,
                self.word_embeddings.scale_grad_by_freq,
                self.word_embeddings.sparse,
            )
            
        
        token_type_embeddings = F.embedding(
                token_type_ids,
                self.token_type_embeddings.weight + self.token_type_embeddings_delta_weight,
                self.token_type_embeddings.padding_idx,
                self.token_type_embeddings.max_norm,
                self.token_type_embeddings.norm_type,
                self.token_type_embeddings.scale_grad_by_freq,
                self.token_type_embeddings.sparse,
            )
    

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = F.embedding(
                    position_ids,
                    self.position_embeddings.weight + self.position_embeddings_delta_weight,
                    self.position_embeddings.padding_idx,
                    self.position_embeddings.max_norm,
                    self.position_embeddings.norm_type,
                    self.position_embeddings.scale_grad_by_freq,
                    self.position_embeddings.sparse,
                )
            
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    

class KNNEnhancedLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, binary_list_dict=None, length_list_dict=None):
        super().__init__(normalized_shape, eps)
        self.binary_list_dict = binary_list_dict
        self.length_list_dict = length_list_dict
        self.binary_weights = None
        self.delta_weight = 0
        self.delta_bias = 0
        
        self.flag = None
    
    def set_delta_by_binary_weight(self):
        if self.binary_weights is None:
            self.delta_weight = 0
            self.delta_bias = 0
            return
        
        if self.flag == "embeddings":
            prefix = ""
        elif self.flag == "output":
            prefix = "output."
        elif self.flag == "attention.output":
            prefix = "attention.output."
        else:
            raise(f"wrong flag:{self.flag}")
            
        device = self.weight.device
        self.delta_weight = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx][f"{prefix}LayerNorm.weight"].to(device) * self.binary_list_dict[idx][f"{prefix}LayerNorm.weight"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)
        if self.bias is not None:
            self.delta_bias = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx][f"{prefix}LayerNorm.bias"].to(device) * self.binary_list_dict[idx][f"{prefix}LayerNorm.bias"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)
        torch.cuda.empty_cache()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight+self.delta_weight, self.bias+self.delta_bias, self.eps)

    
class KNNEnhancedRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config, position_embedding_type=None, binary_list_dict=None, length_list_dict=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.binary_list_dict = binary_list_dict
        self.length_list_dict = length_list_dict
        self.binary_weights = None
        self.delta_query_weight = 0
        self.delta_key_weight = 0
        self.delta_value_weight = 0
        self.delta_query_bias = 0
        self.delta_key_bias = 0
        self.delta_value_bias = 0


    def set_delta_by_binary_weight(self):
        if self.binary_weights is None:
            self.delta_query_weight = 0
            self.delta_key_weight = 0
            self.delta_value_weight = 0
            self.delta_query_bias = 0
            self.delta_key_bias = 0
            self.delta_value_bias = 0
            return

        device = self.query.weight.device
        self.delta_query_weight = torch.sum(torch.stack([
            self.binary_weights[i] * self.length_list_dict[i]["attention.self.query.weight"].to(device) * self.binary_list_dict[i]["attention.self.query.weight"].to(device)
            for i in range(len(self.binary_weights))
        ]), dim=0)
        self.delta_key_weight = torch.sum(torch.stack([
            self.binary_weights[i] * self.length_list_dict[i]["attention.self.key.weight"].to(device) * self.binary_list_dict[i]["attention.self.key.weight"].to(device)
            for i in range(len(self.binary_weights))
        ]), dim=0)
        self.delta_value_weight = torch.sum(torch.stack([
            self.binary_weights[i] * self.length_list_dict[i]["attention.self.value.weight"].to(device) * self.binary_list_dict[i]["attention.self.value.weight"].to(device)
            for i in range(len(self.binary_weights))
        ]), dim=0)
        
        if self.query.bias is not None:
            self.delta_query_bias = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx]["attention.self.query.bias"].to(device) * self.binary_list_dict[idx]["attention.self.query.bias"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)

        if self.key.bias is not None:
            self.delta_key_bias = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx]["attention.self.key.bias"].to(device) * self.binary_list_dict[idx]["attention.self.key.bias"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)

        if self.value.bias is not None:
            self.delta_value_bias = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx]["attention.self.value.bias"].to(device) * self.binary_list_dict[idx]["attention.self.value.bias"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        if isinstance(self.delta_query_weight, torch.Tensor):
            mixed_query_layer = self.query(hidden_states) + torch.matmul(hidden_states, self.delta_query_weight.T) + self.delta_query_bias

            is_cross_attention = encoder_hidden_states is not None
            if is_cross_attention and past_key_value is not None:
                # reuse k,v, cross_attentions
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            elif is_cross_attention:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states) + torch.matmul(encoder_hidden_states, self.delta_key_weight.T) + self.delta_key_bias)
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states) + torch.matmul(encoder_hidden_states, self.delta_value_weight.T) + self.delta_value_bias)
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = self.transpose_for_scores(self.key(hidden_states) + torch.matmul(hidden_states, self.delta_key_weight.T) + self.delta_key_bias)
                value_layer = self.transpose_for_scores(self.value(hidden_states) + torch.matmul(hidden_states, self.delta_value_weight.T) + self.delta_value_bias)
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states) + torch.matmul(hidden_states, self.delta_key_weight.T) + self.delta_key_bias)
                value_layer = self.transpose_for_scores(self.value(hidden_states) + torch.matmul(hidden_states, self.delta_value_weight.T) + self.delta_value_bias)
        else:
            mixed_query_layer = self.query(hidden_states)
            is_cross_attention = encoder_hidden_states is not None
            if is_cross_attention and past_key_value is not None:
                # reuse k,v, cross_attentions
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            elif is_cross_attention:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    
    
    

class KNNEnhancedRobertaSelfOutput(RobertaSelfOutput):
    def __init__(self, config, binary_list_dict=None, length_list_dict=None):
        super().__init__(config)
        self.LayerNorm = KNNEnhancedLayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                              binary_list_dict=binary_list_dict,
                                              length_list_dict=length_list_dict)
        self.LayerNorm.flag = "attention.output"
        
        
        self.binary_list_dict = binary_list_dict
        self.length_list_dict = length_list_dict
        self.binary_weights = None
        self.delta_dense_weight = 0
        self.delta_dense_bias = 0

    def set_delta_by_binary_weight(self):
        if self.binary_weights is None:
            self.delta_dense_weight = 0
            self.delta_dense_bias = 0
            return
        device = self.dense.weight.device
        self.delta_dense_weight = torch.sum(torch.stack([
            self.binary_weights[i] * self.length_list_dict[i]["attention.output.dense.weight"].to(device) * self.binary_list_dict[i]["attention.output.dense.weight"].to(device)
            for i in range(len(self.binary_weights))
        ]), dim=0)
        if self.dense.bias is not None:
            self.delta_dense_bias = torch.sum(torch.stack([weight.to(device) * self.length_list_dict[idx]["attention.output.dense.bias"].to(device) * self.binary_list_dict[idx]["attention.output.dense.bias"].to(device) for idx, weight in enumerate(self.binary_weights)]), dim=0)
        
        self.LayerNorm.binary_weights = self.binary_weights
        self.LayerNorm.set_delta_by_binary_weight()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, self.dense.weight + self.delta_dense_weight, self.dense.bias + self.delta_dense_bias)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class KNNEnhancedRobertaAttention(RobertaAttention):
    def __init__(self, config, binary_list_dict=None, length_list_dict=None):
        super().__init__(config)
        self.self = KNNEnhancedRobertaSelfAttention(config,
                                                    binary_list_dict=binary_list_dict,
                                                    length_list_dict=length_list_dict)
        self.output = KNNEnhancedRobertaSelfOutput(config,
                                                   binary_list_dict=binary_list_dict,
                                                   length_list_dict=length_list_dict)
        self.binary_weights = None

    def set_delta_by_binary_weight(self):
        self.self.binary_weights = self.binary_weights
        self.output.binary_weights = self.binary_weights
        self.self.set_delta_by_binary_weight()
        self.output.set_delta_by_binary_weight()


class KNNEnhancedRobertaIntermediate(RobertaIntermediate):
    def __init__(self, config, binary_list_dict=None, length_list_dict=None):
        super().__init__(config)
        self.binary_list_dict = binary_list_dict
        self.length_list_dict = length_list_dict
        self.binary_weights = None
        self.delta_dense_weight = 0
        self.delta_dense_bias = 0

    def set_delta_by_binary_weight(self):
        if self.binary_weights is None:
            self.delta_dense_weight = 0
            self.delta_dense_bias = 0
            return
        device = self.dense.weight.device
        self.delta_dense_weight = torch.sum(torch.stack([
            self.binary_weights[i] * self.length_list_dict[i]["intermediate.dense.weight"].to(device) * self.binary_list_dict[i]["intermediate.dense.weight"].to(device)
            for i in range(len(self.binary_weights))
        ]), dim=0)
        if self.dense.bias is not None:
            self.delta_dense_bias = torch.sum(torch.stack([
                self.binary_weights[i] * self.length_list_dict[i]["intermediate.dense.bias"].to(device) * self.binary_list_dict[i]["intermediate.dense.bias"].to(device)
                for i in range(len(self.binary_weights))
            ]), dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, self.dense.weight + self.delta_dense_weight, self.dense.bias + self.delta_dense_bias)
        return self.intermediate_act_fn(hidden_states)


class KNNEnhancedRobertaOutput(RobertaOutput):
    def __init__(self, config, binary_list_dict=None, length_list_dict=None):
        super().__init__(config)
        self.LayerNorm = KNNEnhancedLayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                              binary_list_dict=binary_list_dict,
                                              length_list_dict=length_list_dict)
        self.LayerNorm.flag = "output"
        
        
        self.binary_list_dict = binary_list_dict
        self.length_list_dict = length_list_dict
        self.binary_weights = None
        self.delta_dense_weight = 0
        self.delta_dense_bias = 0

    def set_delta_by_binary_weight(self):
        if self.binary_weights is None:
            self.delta_dense_weight = 0
            self.delta_dense_bias = 0
            return
        device = self.dense.weight.device
        self.delta_dense_weight = torch.sum(torch.stack([
            self.binary_weights[i] * self.length_list_dict[i]["output.dense.weight"].to(device) * self.binary_list_dict[i]["output.dense.weight"].to(device)
            for i in range(len(self.binary_weights))
        ]), dim=0)
        if self.dense.bias is not None:
            self.delta_dense_bias = torch.sum(torch.stack([
                self.binary_weights[i] * self.length_list_dict[i]["output.dense.bias"].to(device) * self.binary_list_dict[i]["output.dense.bias"].to(device)
                for i in range(len(self.binary_weights))
            ]), dim=0)
        
        self.LayerNorm.binary_weights = self.binary_weights
        self.LayerNorm.set_delta_by_binary_weight()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, self.dense.weight + self.delta_dense_weight, self.dense.bias + self.delta_dense_bias)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class KNNEnhancedRobertaLayer(RobertaLayer):
    def __init__(self, config, binary_list_dict=None, length_list_dict=None):
        super().__init__(config)
        self.attention = KNNEnhancedRobertaAttention(config,
                                                     binary_list_dict=binary_list_dict,
                                                     length_list_dict=length_list_dict)
        self.intermediate = KNNEnhancedRobertaIntermediate(config,
                                                           binary_list_dict=binary_list_dict,
                                                           length_list_dict=length_list_dict)
        self.output = KNNEnhancedRobertaOutput(config,
                                               binary_list_dict=binary_list_dict,
                                               length_list_dict=length_list_dict)
        self.binary_weights = None

    def set_delta_by_binary_weight(self):
        self.attention.binary_weights = self.binary_weights
        self.intermediate.binary_weights = self.binary_weights
        self.output.binary_weights = self.binary_weights
        self.attention.set_delta_by_binary_weight()
        self.intermediate.set_delta_by_binary_weight()
        self.output.set_delta_by_binary_weight()


class KNNEnhancedRobertaEncoder(RobertaEncoder):
    def __init__(self, config, binary_dict_list_dict=None, length_dict_list_dict=None):
        super().__init__(config)
        
        self.layer = nn.ModuleList([
            KNNEnhancedRobertaLayer(config,
                                    binary_list_dict=binary_dict_list_dict[f"roberta.encoder.layer.{i}"],
                                    length_list_dict=length_dict_list_dict[f"roberta.encoder.layer.{i}"])
            for i in range(config.num_hidden_layers)
        ])
        self.binary_weights = None

    def set_delta_by_binary_weight(self):
        for layer in self.layer:
            layer.binary_weights = self.binary_weights
            layer.set_delta_by_binary_weight()


            
            
            
            
            
            
class KNNEnhancedRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True, binary_dict_list_dict=None, length_dict_list_dict=None):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = KNNEnhancedRobertaEmbeddings(
            config,
            binary_list_dict=binary_dict_list_dict["roberta.embeddings"],
            length_list_dict=length_dict_list_dict["roberta.embeddings"]
        )
        
        self.encoder = KNNEnhancedRobertaEncoder(
            config,
            binary_dict_list_dict=binary_dict_list_dict,
            length_dict_list_dict=length_dict_list_dict
        )
        self.binary_weights = None

    def set_delta_by_binary_weight(self, use_delta):
        if use_delta:
            assert self.binary_weights is not None
            self.embeddings.binary_weights = self.binary_weights
            self.encoder.binary_weights = self.binary_weights
            self.embeddings.set_delta_by_binary_weight()
            self.encoder.set_delta_by_binary_weight()
        else:
            self.embeddings.binary_weights = None
            self.encoder.binary_weights = None
            self.embeddings.set_delta_by_binary_weight()
            self.encoder.set_delta_by_binary_weight()
        torch.cuda.empty_cache()
