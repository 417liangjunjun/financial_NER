# 金融领域的命名实体识别

说明本项目CCF比赛的一个项目，详情可以看[这里](https://www.datafountain.cn/competitions/353)

## 实现思路

将问题转化为一个序列标注问题，去预测每个实体的开始和结束,利用Bert+CRF的形式进行实体识别

## 代码结构
```
├─data
│  └─utils
├─library \\Allennlp所需
│  ├─config 
│  ├─dataset_readers
│  ├─models
│  └─predictor
├─preprocessor  \\用于数据前处理
└─processor \\最后用于实体识别的类
```

## 数据处理
利用`preprocess/`下的文件对数据进行了清理，`preprocess.py`做的是去除一些网页符号，全角转半角这些功能。`data/`下的`.csv`是原始数据
，`train.txt`是经过处理后的训练数据。`data_augmentor.py`是做的是数据增加，会将一定比例将句子中一些实体替换成随机生成的字符串用于增加数据量
以及让模型更好的捕捉到实体的上下文信息，而不是通过实体本身去识别实体。`training_data_generator.py`将txt文件转化成
给模型训练用的json文件

## 运行脚本

### 下载bert模型
点击[这里](https://pan.baidu.com/s/1KBNNygpDlLeO7dvKB79zTg)(提取码aup8)
下载预训练好的bert模型。

### 修改配置文件  
训练数据地址，bert模型地址，训练器，学习率等参数均写在`library/config/bert_crf_tagger.jsonnet`中
,需要根据实际情况对其进行修改

### 还可以做的
模型框架应该不用调了  
####数据处理
数据前后处理可以再仔细做一下，目前的训练数据仍然没有洗干净。
另外预测生成的实体中有很多单字的实体或者一些其他可以通过增加规则去掉的badcase。

#### 评价指标
我没看比赛要求的评价指标，可以看下官网的评价指标，评测一下我们的效果跟排行榜比较一下
