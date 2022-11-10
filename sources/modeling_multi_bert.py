import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable

from transformers.modeling_bert import BertPreTrainedModel, BertModel, BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """Bert Model with multi tasks heads on top (linear layers on top of
    the hidden-states output) e.g. for Part-of-Speech (PoS) tagging and Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
)
class MTDNN(BertPreTrainedModel):
    def __init__(self, config, auxiliary_num_labels_list, main_num_labels):
        super().__init__(config)
        self.auxiliary_num_labels_list = auxiliary_num_labels_list
        self.main_num_labels = main_num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.auxiliary_classifier_list = []
        for x in range(len(auxiliary_num_labels_list)):
            auxiliary_classifier = nn.Linear(config.hidden_size, auxiliary_num_labels_list[x])
            self.auxiliary_classifier_list.append(auxiliary_classifier)
        self.auxiliary_classifier_list = nn.ModuleList(self.auxiliary_classifier_list)
        self.main_classifier = nn.Linear(config.hidden_size, main_num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_id=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        if task_id[0] == len(self.auxiliary_classifier_list):
            logits = self.main_classifier(sequence_output)
            num_labels = self.main_num_labels
        else:
            x = int(task_id[0])
            auxiliary_classifier = self.auxiliary_classifier_list[x]
            logits = auxiliary_classifier(sequence_output)
            num_labels = self.auxiliary_num_labels_list[x]

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
