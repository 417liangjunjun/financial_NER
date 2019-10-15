# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: extractor.py
@time: 2019/9/21 22:03
@desc: 
"""
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import import_submodules
from preprocessor.preprocess import normalize_string
import csv
from tqdm import tqdm
import_submodules('library')


class ArticleTypeClassifier(object):
    def __init__(self, model_path, answer_path='./result.csv', cuda_device=-1):
        archive = load_archive(model_path, cuda_device=cuda_device)
        self.predictor = Predictor.from_archive(archive, 'bert_crf_predictor')
        self.answer_path = answer_path

    def predict(self, text):
        input = {}
        input['text'] = text
        return self.predictor.predict_json(input)

    def predict_text(self, title, content):
        text_list = []
        if len(content) + len(title) < 500:
            text = title + content
            text_list.append(text)
        # 如果title＋content长度大于500,那么找到满足title+content
        # 长度小于500条件下，使得content长度最大的句号将content分开,并分别拼接title
        elif len(content) > 500 - len(title):
            content_list = content.split('。')
            text = title
            for _content in content_list:
                if len(title + '。' + _content) > 500:
                    temp = title + '。' + _content
                    text_list.append(temp[:500])
                elif len(text + '。' + _content) > 500:
                    text_list.append(text)
                    text = title + '。' + _content
                else:
                    text = text + '。' + _content
        answer_list = []
        for text in text_list:
            answer = self.predict(text)
            answer = [ans[0] for ans in answer if ans]
            answer_list.extend(answer)
        return set(answer_list)

    def predict_csv(self, file_path):
        with open(file_path) as fin, open(self.answer_path, 'w') as csv_file:
            data_list = csv.reader(fin)
            for id, title, content in tqdm(data_list):
                title = normalize_string(title)
                content = normalize_string(content)
                text = title + content
                answer = ''
                if text:
                    print(id,title,content)
                    answer = self.predict_text(title, content)
                answer = ';'.join([ans for ans in list(answer)])
                csv_file.write(id + ',' + answer+'\n')
                # print(title, content)
                # print(id,answer)
                # print('\n')


if __name__ == "__main__":
    article_type_classifier = ArticleTypeClassifier(
        model_path='/home/liangjj/fin_ner/model/0921/model.tar.gz',
        cuda_device=2)
    # print(article_type_classifier.predict_text('','专项行动中,各地文化市场综合执法机构严查含有宣扬赌博内容的网络游戏,北京市、天津市、重庆市等地查办了北京联众互动网络股份有限公司、天津网狐信息科技有限公司、重庆虎阿网络科技有限公司提供含有宣扬赌博内容的网络游戏产品案;严查含有低俗内容的网络游戏,针对近期社会影响恶劣的模拟当官类游戏,北京市、湖北省、海南省等地查办了北京六趣网络科技有限公司、武汉爪游互娱科技有限公司、海南游情网络科技有限公司提供含有违背社会公德内容的网络游戏产品案,责令官居几品全民宫斗老爷吉祥等游戏进行整改;查处网络游戏未要求实名注册、以随机抽取等偶然方式诱导消费等违规经营行为,江苏省查办了南京雪糕网络科技有限公司未要求网络游戏用户使用有效身份证件进行实名注册案,北京市查办了北京畅聊天下科技股份有限公司以随机抽取等偶然方式诱导网络游戏用户采取投入法定货币或者网络游戏虚拟货币方式获取网络游戏产品和服务案;严肃查处恶搞红色经典及英雄人物视频,针对恶搞黄河大合唱等红色经典及英雄人物视频的问题,文化和旅游部追根溯源,指导四川省文化市场执法监督局和成都市文化市场综合执法总队开展查处工作,严查恶搞视频的源头制作公司,成都市文化市场综合执法总队依法给予四川盛世天府传媒有限公司警告和罚款的高限处罚;严查含有禁止内容的网络表演,持续加强网络表演市场监管执法,福建省、广东省等地查办了福建省灵朵网络科技有限公司、厦门远征网络科技有限公司、深圳大鸟网络科技有限公司提供含有宣扬淫秽、宣扬赌博、危害社会公德等禁止内容的网络表演案,依法查处一兔秀场天天乐直播供享直播等网络表演平台'))
    article_type_classifier.predict_csv('/home/liangjj/fin_ner/data/Test_Data.csv')
