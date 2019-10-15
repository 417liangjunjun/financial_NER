# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: bert-crf.py
@time: 2019/9/21 21:11
@desc: 
"""
from typing import List
import logging
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register('bert_crf_predictor')
class BertCrfPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(text)

    @overrides
    def predict_json(self, input: JsonDict) -> JsonDict:
        input_instance = self._json_to_instance(input)
        out_dict = self.predict_instance(input_instance)
        result = out_dict['tags']
        answer = ""
        start = -1
        answers = []
        text = input['text']
        for i, lable in enumerate(result):
            if lable == "B":
                if answer != "" and start != -1:
                    answers.append([answer, start])
                answer = ""
                answer += text[i]
                start = i
            elif lable == "I":
                answer += text[i]
            else:
                continue
        if answer != "" and start != -1:
            answers.append([answer, start])
        return answers
