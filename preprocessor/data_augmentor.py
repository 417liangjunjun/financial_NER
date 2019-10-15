# -*- coding: UTF-8 -*-
import os
from collections import defaultdict, namedtuple
import json
import random

FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

news = namedtuple('news', ['title', 'content', 'entity'])


class DataAugementor(object):
    def __init__(self, file_path=os.path.join(FILE_PATH, 'data/train.txt'),
                 out_file_path=os.path.join(FILE_PATH, 'data/augement_train.txt')):
        self.file_path = file_path
        self.out_file_path = out_file_path
        self.char_set = set()

    def read_data(self):
        with open(self.file_path, encoding='utf-8') as f:
            data_list = f.readlines()
        data_dict = {}
        temp_list = []
        min_length = 9999
        max_length = 0
        entity_length_dict = defaultdict(int)
        char_list = []
        for data in data_list[1:]:
            id, title, content, entity = data.split('\t')
            # 去掉末尾\n
            entity_list = entity.strip().split(';')
            entity_list.sort(key=lambda x: -len(x))
            temp_list.extend(entity_list)
            bad_flag = False
            for _entity in entity_list:
                if len(_entity) >30:
                    bad_flag = True
                    continue
                if not _entity:
                    entity_list.remove(_entity)
                    continue
                entity_length_dict[len(_entity)] += 1
                char_list.extend(_entity)
                min_length = min(len(_entity), min_length)
                max_length = max(len(_entity), max_length)
            if not bad_flag:
                data_dict[id] = {'id': id, 'title': title, 'content': content, 'entity': entity_list, 'real': 1}
        char_set = set(char_list)
        return data_dict, char_set, entity_length_dict

    def augement(self, fake_num=4):
        data_dict, char_set, entity_length_dict = self.read_data()
        length_list = []
        char_list = list(char_set)
        fake_data_dict = defaultdict(list)
        for length, num in entity_length_dict.items():
            if length >30:
                print(length,num)
            length_list.extend([length] * num)
        for id, item_dict in data_dict.items():
            # 每条数据生成fake_num条人造数据。
            for i in range(fake_num):
                _, title, content, entity_list, _ = list(item_dict.values())
                fake_entity_list = []
                for entity in entity_list:
                    entity_length = random.choice(length_list)
                    fake_entity = ''
                    for j in range(entity_length):
                        fake_entity += random.choice(char_list)
                    fake_entity_list.append(fake_entity)
                    title = title.replace(entity, fake_entity)
                    content = content.replace(entity, fake_entity)
                fake_new = {'id': id, 'title': title, 'content': content, 'entity': fake_entity_list, 'real': 0}
                fake_data_dict[id].append(fake_new)
        with open(self.out_file_path, 'w') as fout:
            for id in data_dict:
                string = json.dumps(data_dict[id], ensure_ascii=False)
                fout.write(string)
                fout.write('\n')
                # print(string)
                for fake_new in fake_data_dict[id]:
                    string = json.dumps(fake_new, ensure_ascii=False)
                    fout.write(string)
                    fout.write('\n')
                    # print(string)


if __name__ == "__main__":
    augementor = DataAugementor()
    augementor.augement()
