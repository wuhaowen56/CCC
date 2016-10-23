# -*- coding: utf-8 -*-
# @Time    : 2016/10/19 23:24
# @Author  : Aries
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm Community Edition

import jieba
import jieba.analyse
import kafang1
import TF
import IDF

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def preprocess():
    #读入停词表、构建停词字典
    stoptxt = open('stopwords.txt')
    word = []
    for line in stoptxt:
        word.append(line.decode('gbk','ignore').encode('utf-8').strip('\n'))
    stopwords = {}.fromkeys([line.rstrip() for line in word])

    #训练集分词、去停词
    accumulate = []

    f = open('test.TRAIN','r')
    new_item = {}
    for line in f:
        accumulate = []
        #编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
        single_line = line.decode('gbk','ignore').encode('utf-8').split('\t')

        userId = single_line[0]
        age = single_line[1]
        gender = single_line[2]

        education = single_line[3]
        key_words = single_line[4:]

        user_word = []
        for item in key_words:
            print '1'
            seg_list = jieba.cut(item)
            for check in seg_list:
                check = check.encode('utf-8')
                if check not in stopwords:
                    user_word.append(check)
                    if single_line[2] != '0':
                        accumulate.append(check)
        _item = {}
        _item.setdefault(gender,accumulate)
        new_item.setdefault(userId,_item)
        #userid gender accumulate

    #去重后的所有单词男女分开
    de_repeat_table = {}
    tem_label_male =[]
    tem_label_female = []


    for userId, _item in new_item.iteritems():
        for label , user_word in _item.iteritems():
            if label == '1' :
                tem_label_male.append(user_word)
            else:
                tem_label_female.append(user_word)

    tem_label1 = [x for x in tem_label_male if tem_label_male.count(x) == 1]
    tem_label2 = [x for x in tem_label_female if tem_label_female.count(x) == 1]

    de_repeat_table.setdefault('1',tem_label1)
    de_repeat_table.setdefault('2', tem_label2)
    return (new_item,de_repeat_table)

k ,l = preprocess()
feature = kafang1.Chi_Square(k,l)
tf1 = TF.tf(k)
idf1 = IDF.idf(tf1)
IDF.tf_idf(tf1, idf1, feature)

