# -*- coding: UTF-8 -*-
# @Time    : 2016/10/7 16:29
# @Author  : Aries
# @Site    : 
# @File    : IDF.py
# @Software: PyCharm Community Edition

import math


# 数据格式  userId word TF  word TF
allTF = {'user1': {'柔和': 0.123, '格式': 0.123, '哈达': 0.12}, 'user2': {'柔和1': 0.123, '格式1': 0.123, '哈达': 0.12}}

userCount = len(allTF)
dict_resIDF = {}
for k, v in allTF.iteritems():
    for k1, v1 in v.iteritems():
        if k1 in dict_resIDF.keys():
            dict_resIDF[k1] += 1
        else:
            dict_resIDF[k1] = 1
for k, v in dict_resIDF.iteritems():
    dict_resIDF[k] = math.log10(v/userCount+0.01)

# do TF-IDF
# 如果特征词袋包含则计算。
feature = {"柔和": 1, "格式": 2, "哈达": 3}
TF_IDF = {}

for k, v in allTF.iteritems():
    for k1, v1 in v.iteritems():
        if k1 in dict_resIDF.keys():
            if k1 in feature:
                TF_IDF[k1] = v1 * dict_resIDF.get(k1)
for k, v in TF_IDF.iteritems():
    print k, v


