# -*- coding: UTF-8 -*-
import os
import csv
import re
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')


def full2half(text: str) -> str:
    res_text = ''
    for char in text:
        cp = ord(char)
        if cp == 0x3000:
            cp = 32
        elif 0xFF01 <= cp <= 0xFF5E:
            cp -= 0xfee0
        new_char = chr(cp)
        res_text += new_char
    return res_text

def normalize_string(string):
    space_reg = re.compile(r'\t|\r|\s|\f')
    string = full2half(string)
    string = space_reg.sub(' ',string)
    string = string.split(' ')
    string = ' '.join([_str for _str in string if _str])
    string = string.split('?')
    string = '?'.join([_str for _str in string if _str])
    string = string.split('!')
    string = '!'.join([_str for _str in string if _str])
    html_reg = re.compile(r'<.*?>|\{?IMG:\d\d?\}?|[^\w,\。\.:\(\)\?!#、;"]')
    string = html_reg.sub('',string)
    return string

def processor_test_text(file_path):
    with open(file_path) as f:
        data_list = csv.reader(f)
        for id, title, content in data_list:
            title = normalize_string(title)
            content = normalize_string(content)

def processor_train_text(file_path):
    with open(file_path) as f,open(os.path.join(os.path.dirname(file_path),'train.txt'),'w') as fout:
        data_list = csv.reader(f)
        for id, title, content,entity in data_list:
            title = normalize_string(title)
            content = normalize_string(content)
            entity = normalize_string(entity)
            fout.write(id+'\t'+title+'\t'+content+'\t'+entity+'\n')

if __name__ =="__main__":
    file_path = os.path.join(FILE_PATH,'data/Train_Data.csv')
    processor_train_text(file_path)