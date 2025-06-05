from transformers import RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaClassificationHead
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from knn_enhanced_roberta_model import KNNEnhancedRobertaModel


class KNNEnhancedRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, binary_dict_list_dict=None, length_dict_list_dict=None, knn=None, dataset_names=None):  # , pretrained_state_dict=None
        super().__init__(config)
        self.roberta = KNNEnhancedRobertaModel(config, add_pooling_layer=False, binary_dict_list_dict=binary_dict_list_dict, length_dict_list_dict=length_dict_list_dict)  # KNNEnhancedRobertaModel

        self.knn = knn
        self.dataset_names = dataset_names
        
        # self.use_knn_enhancement = False
        self.use_knn_enhancement = knn is not None and binary_dict_list_dict is not None and length_dict_list_dict is not None
        
    def compute_binary_weights(self, input_features):
        with torch.no_grad():
            knn_probs = self.knn.predict_proba(input_features.cpu())  # shape: [batch_size, num_tasks]
            knn_weights_mean = {dataset: 0.0 for dataset in self.dataset_names}
            for sample_probs in knn_probs:
                for i, dataset in enumerate(self.dataset_names):
                    knn_weights_mean[dataset] += sample_probs[i]
            for dataset in knn_weights_mean:
                knn_weights_mean[dataset] /= len(knn_probs)
        return torch.tensor(list(knn_weights_mean.values()), dtype=torch.float32, device=input_features.device)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        torch.cuda.empty_cache()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            self.roberta.set_delta_by_binary_weight(use_delta=False)
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]

            if self.use_knn_enhancement:                
                feature_batch = sequence_output[:, 0, :]
                weight = self.compute_binary_weights(feature_batch)                
                max_idx = torch.argmax(weight)
                self.roberta.binary_weights = torch.zeros_like(weight)
                self.roberta.binary_weights[max_idx] = 1.0
                self.roberta.set_delta_by_binary_weight(use_delta=True)
                
                outputs = self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                sequence_output = outputs[0]
                self.roberta.set_delta_by_binary_weight(use_delta=False)


            logits = self.classifier(sequence_output)


            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output
        torch.cuda.empty_cache()
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
