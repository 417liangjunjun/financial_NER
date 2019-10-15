# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: bert_crf_tagger.py
@time: 2019/9/8 13:57
@desc: 
"""
import logging
from typing import Dict, List, Optional, Any
import torch
from torch.nn.modules.linear import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from overrides import overrides
import allennlp.nn.util as util
logger = logging.getLogger(__name__)


@Model.register("bert_crf_tagger")
class BertCrfTaggerModel(Model):
    """
    This class implements BERT for ReadingComprehension using HuggingFace implementation

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: Optional[float] = 0,
                 label_encoding: Optional[str] = 'BIO',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        """
        :param vocab: ``Vocabulary``
        :param text_field_embedder: ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
        :param dropout:
        :param label_encoding: BIO
        :param initializer:``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
        :param regularizer:``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
        """
        super(BertCrfTaggerModel, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size('labels')

        self._labels_predictor = Linear(self._text_field_embedder.get_output_dim(), self.num_tags)
        self.dropout = torch.nn.Dropout(dropout)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self._f1_metric = SpanBasedF1Measure(vocab, tag_namespace='labels', label_encoding=label_encoding)
        labels = self.vocab.get_index_to_token_vocabulary('labels')
        constraints = allowed_transitions(label_encoding, labels)
        self.label_to_index = self.vocab.get_token_to_index_vocabulary('labels')
        self.crf = ConditionalRandomField(self.num_tags, constraints, include_start_end_transitions=False)
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                labels: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : Dict[str, torch.LongTensor]
            From a ``TextField``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
        labels: A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : Tensor, a linear transformation to the incoming data: :math:`y = xA^T + b`
        mask : mask
        tags : int
            probabilities of the span end position (inclusive).
        probabilities : torch.FloatTensor
            The result of ``softmax(tag_logits, dim=-1)``.
        loss : torch.IntTensor
        """

        mask = util.get_text_field_mask(text)
        embedded_text = self._text_field_embedder(text)


        tag_logits = self._labels_predictor(embedded_text)
        # tag_logits = util.replace_masked_values(tag_logits, passage_mask.unsqueeze(-1), -1e32)
        predicted_probability = torch.nn.functional.softmax(tag_logits, dim=-1)
        best_paths = self.crf.viterbi_tags(tag_logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]
        output_dict = {"logits": tag_logits, "mask": mask, "tags": predicted_tags,
                       "probabilities": predicted_probability}

        # Compute the loss for training.
        if labels is not None:
            log_likelihood = self.crf(tag_logits, labels, mask)
            output_dict["loss"] = -log_likelihood

            class_probabilities = tag_logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    if tag_id >= len(self.label_to_index):
                        tag_id = self.label_to_index['O']
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, mask.float())
            self._f1_metric(class_probabilities, labels, mask.float())

        if metadata is not None:
            text_tokens = []
            for i, _ in enumerate(metadata):
                text_tokens.append(metadata[i].get('question_tokens', []))
            output_dict['text_tokens'] = text_tokens
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace="labels")
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """获取评价指标"""
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        f1_dict = self._f1_metric.get_metric(reset=reset)

        metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return


