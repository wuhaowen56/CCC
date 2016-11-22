# -*- coding: utf-8 -*-
# @Time    : 2016/10/19 23:24
# @Author  : Aries
# @Site    :
# @File    : preprocess.py
# @Software: PyCharm Community Edition

from sklearn import svm
import jieba
import jieba.analyse
import kafang1
import TF
import IDF
import cPickle as pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
import numpy as np
from sklearn import metrics

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from time import time
import re
import jieba
import jieba.analyse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def readbunchobj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch


def writebunchobj(path, bunchobj):
    file_obj = open(path, "wb")
    pickle.dump(bunchobj, file_obj)
    file_obj.close()


def preprocess(path):
    #读入停词表、构建停词字典
    stoptxt = open('stopwords.txt')
    word = []
    for line in stoptxt:
        word.append(line.decode('gb18030','ignore').encode('utf-8').strip('\n'))
    stopwords = {}.fromkeys([line.rstrip() for line in word])

    new_table = []
    f = open(path,'r')
    for line in f:
        #编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
        single_line = line.decode('gbk','ignore').encode('utf-8').split()
        key_words = single_line[4:]
        user_word = ''
        for item in key_words:
            seg_list = jieba.cut(item)
            for check in seg_list:
                check = check.encode('utf-8')
                if  single_line[3]!='0':
                    user_word = user_word + ' '+check
        if single_line[3]!='0':
            new_item = [single_line[0],single_line[3],user_word]
            new_table.append(new_item)
    #

    bunch = Bunch(label=[], userid=[], contents=[])
    for k in new_table:
        bunch.userid.append(k[0])
        bunch.label.append(k[1])
        bunch.contents.append(k[2])
    return bunch


def preprocess1(path):
    # 读入停词表、构建停词字典
    stoptxt = open('stopwords.txt')
    word = []
    for line in stoptxt:
        word.append(line.decode('gb18030', 'ignore').encode('utf-8').strip('\n'))
    stopwords = {}.fromkeys([line.rstrip() for line in word])

    new_table = []
    f = open(path, 'r')
    for line in f:
        # 编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
        single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
        key_words = single_line[1:]
        user_word = ''
        for item in key_words:
            seg_list = jieba.cut(item)
            for check in seg_list:
                check = check.encode('utf-8')
                user_word = user_word + ' ' + check
        new_item = [single_line[0], user_word]
        new_table.append(new_item)
    bunch = Bunch(label=[], userid=[], contents=[])
    for k in new_table:
        bunch.userid.append(k[0])
        bunch.label.append('1')# 全部设为1
        bunch.contents.append(k[1])
    return bunch

def writebunchobj(path, bunchobj):
    file_obj = open(path, "wb")
    pickle.dump(bunchobj, file_obj)
    file_obj.close()
data_train  = preprocess('train_word_bag/user_tag_query.10W.TRAIN')
# data_train = readbunchobj('seg_edu_train.txt')
writebunchobj('seg_edu_train.txt', data_train)
data_test  = preprocess1('test_word_bag/user_tag_query.10W.TEST')
writebunchobj('seg_edu_test.txt', data_test)
# data_test = readbunchobj('seg_edu_test.txt')
y_train, y_test = data_train.label, data_test.label
def get_stop_words():
    result = set()
    for line in open('stopwords.txt', 'r').readlines():
        result.add(line.strip())
    return result
stop_words = get_stop_words()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words=stop_words)
X_train = vectorizer.fit_transform(data_train.contents)

X_test = vectorizer.transform(data_test.contents)

ch2 = SelectKBest(chi2, k=50000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

# SVM, transform into probability output
from sklearn.calibration import CalibratedClassifierCV

clf_ = svm.LinearSVC(C=0.15) #still the best when C=0.2

# clf_ = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.01,penalty='l1',dual=False))
clf_SVM = CalibratedClassifierCV(clf_)

# predict
clf_SVM.fit(X_train,y_train)
predicted = clf_SVM.predict(X_test)
total = len(predicted)
print total
rate = 0
f = open('resoult_edu_1.txt','wb')

for flabel, file_name, expct_cate in zip(data_test.label, data_test.userid, predicted):
    # print file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate
    f.write(expct_cate)
    f.write('\n')
    if flabel != expct_cate:
        rate += 1
f.close()
# 精度
print "error rate:", float(rate) * 100 / float(total), "%"
print "edu预测完毕!!!"


