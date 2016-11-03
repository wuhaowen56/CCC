# -*- coding: utf-8 -*-
# @Time    : 2016/10/19 23:24
# @Author  : Aries
# @Site    :
# @File    : preprocess.py
# @Software: PyCharm Community Edition

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
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
import  re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
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

# def metrics_result(actual, predict):
#     print '精度:{0:.3f}'.format(metrics.precision_score(actual, predict))
#     print '召回:{0:0.3f}'.format(metrics.recall_score(actual, predict))
#     print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict))


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
                if check not in stopwords and single_line[1]!='0':
                    user_word = user_word + ' '+check
        if single_line[1]!='0':
            new_item = [single_line[0],single_line[1],user_word]
            new_table.append(new_item)

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
                if check not in stopwords:
                    user_word = user_word + ' ' + check
        new_item = [single_line[0], user_word]
        new_table.append(new_item)
    bunch = Bunch(label=[], userid=[], contents=[])
    for k in new_table:
        bunch.userid.append(k[0])
        bunch.label.append('1')# 全部设为1
        bunch.contents.append(k[1])
    return bunch


def readbunchobj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch

def writebunchobj(path, bunchobj):
    file_obj = open(path, "wb")
    pickle.dump(bunchobj, file_obj)
    file_obj.close()
# data_train  = preprocess('train_word_bag/user_tag_query.2W.TRAIN')
data_train = readbunchobj('seg_age_train.txt')
# writebunchobj('seg_age_train.txt', data_train)
# data_test  = preprocess1('test_word_bag/user_tag_query.2W.TEST')
data_test = readbunchobj('seg_age_test.txt')
# writebunchobj('seg_age_test.txt', data_test)
y_train, y_test = data_train.label, data_test.label

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.contents)

print('-----------------------------2')
X_test = vectorizer.transform(data_test.contents)

ch2 = SelectKBest(chi2, k=40000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
# tfidfspace = Bunch( label=bunch.label, userid=bunch.userid, tdm=[], vocabulary={})
#
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
# transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
# # 文本转为词频矩阵,单独保存字典文件
# tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
# tfidfspace.vocabulary = vectorizer.vocabulary_

# 创建词袋的持久化
# space_path = "train_word_bag/tfdifspace_edu_1.dat"  # 词向量空间保存路径
# writebunchobj(space_path, tfidfspace)

print "if-idf词向量空间创建成功！！！"

# tfidfspace1 = Bunch( label=bunch1.label, userid=bunch1.userid, tdm=[], vocabulary={})
# vectorizer1 = TfidfVectorizer(vocabulary = vectorizer.vocabulary_)
# transformer1 = TfidfTransformer( )  # 该类会统计每个词语的tf-idf权值
# # 文本转为词频矩阵,单独保存字典文件
# tfidfspace1.tdm = vectorizer1.fit_transform(bunch1.contents)

# 创建词袋的持久化
# space_path = "test_word_bag/tfdifspace_edu_1.dat"  # 词向量空间保存路径
# writebunchobj(space_path, tfidfspace1)
#
# trainpath = "train_word_bag/tfdifspace_edu_1.dat"
# train_set = readbunchobj(trainpath)
#
#
# testpath = "test_word_bag/tfdifspace_edu_1.dat"
# test_set = readbunchobj(testpath)

model = svm.LinearSVC(C=0.35)

# model = GradientBoostingClassifier()

model.fit(X_train, y_train)
predicted = model.predict(X_test)
total = len(predicted)
print total
rate = 0

f = open('resoult_age_11.txt','wb')
for flabel, file_name, expct_cate in zip(data_test.label, data_test.userid, predicted):
    # print file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate
    f.write(expct_cate)
    f.write('\n')
    if flabel != expct_cate:
        rate += 1
f.close()
# 精度
print "error rate:", float(rate) * 100 / float(total), "%"
print "age预测完毕!!!"

