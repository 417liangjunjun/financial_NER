# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: training_data_generator.py
@time: 2019/9/15 14:15
@desc: 生成训练数据
"""
import os
import json
import re
import random
import json
from tqdm import tqdm

FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


# class Answer (object):
#

class TrainingDataGenerator(object):
    def __init__(self, file_path=os.path.join(FILE_PATH, 'data/augement_train.txt'),
                 train_data_path=os.path.join(FILE_PATH, 'data/final_train.txt'),
                 dev_data_path=os.path.join(FILE_PATH, 'data/final_dev.txt'),
                 test_data_path=os.path.join(FILE_PATH, 'data/final_test.txt')):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.dev_data_path = dev_data_path
        self.data_path = file_path

    def read_data(self):
        final_data = []
        with open(self.data_path) as f:
            data_list = f.readlines()
            for data in tqdm(data_list):
                data = json.loads(data)
                final_data.extend(self.trans_data(data))
        return final_data

    def trans_data(self, data):
        title = data['title']
        content = data['content']
        entity_list = data['entity']
        real = data['real']
        id = data['id']
        final_data = []
        if len(content) + len(title) < 500:
            text = title + content
            answer = self.find_entity_index(text, entity_list)
            final_data.append({'id': id, 'text': text, 'entity': answer, 'real': real})
        # 如果title＋content长度大于500,那么找到满足title+content
        # 长度小于500条件下，使得content长度最大的句号将content分开,并分别拼接title
        elif len(content) > 500 - len(title):
            content_list = content.split('。')
            text = title
            for _content in content_list:
                if len(title + '。' + _content) > 500:
                    continue
                if len(text + '。' + _content) > 500:
                    answer = self.find_entity_index(text, entity_list)
                    final_data.append({'id': id, 'text': text, 'entity': answer, 'real': real})
                    text = title + '。' + _content
                else:
                    text = text + '。' + _content
            answer = self.find_entity_index(text, entity_list)
            final_data.append({'id': id, 'text': text, 'entity': answer, 'real': real})
        return final_data

    def find_entity_index(self, text, entity_list):
        answer = []
        entity_list.sort(key=lambda x: -len(x))
        text_flag = [0] * len(text)
        for entity in entity_list:
            if not entity:
                continue
            start_index = 0
            temp_text = text
            while (temp_text.find(entity) >= 0):
                if not text_flag[temp_text.find(entity) + start_index]:
                    answer.append((entity, temp_text.find(entity) + start_index))
                for i in range(temp_text.find(entity) + start_index,
                               temp_text.find(entity) + start_index + len(entity)):
                    text_flag[i] = 1
                start_index += temp_text.find(entity) + len(entity)
                temp_text = temp_text[temp_text.find(entity) + len(entity):]
        return answer

    def generate_data(self):
        final_data = self.read_data()
        total_num = len(final_data)
        train_data = final_data[:int(0.9 * total_num)]
        dev_data = [data for data in final_data[int(0.9 * total_num):] if data['real']]
        # 暂时test_data
        test_data = []
        self.dumps(train_data, self.train_data_path)
        self.dumps(dev_data, self.dev_data_path)
        self.dumps(test_data, self.test_data_path)

    def dumps(self, final_data, file_path):
        random.shuffle(final_data)
        with open(file_path, 'w') as f:
            for data in final_data:
                f.write(json.dumps(data, ensure_ascii=False))
                f.write('\n')


if __name__ == "__main__":
    generator = TrainingDataGenerator()
    generator.generate_data()
