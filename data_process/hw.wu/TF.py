# -*- coding: utf-8 -*-
# @Time    : 2016/10/7 15:29
# @Author  : Aries
# @Site    : 
# @File    : TF.py
# @Software: PyCharm Community Edition

from __future__ import division

# tf 是该词在本篇中的概率,每一篇query分开处理
cutwords = ['柔和','双沟','女生',	'中财网','首页', '财经','柔和']
# cutwords 是每一篇分词后的结果

# 计算nomalTF


dict_resTF = {}
for word in cutwords:
    if word in dict_resTF.keys():
        dict_resTF[word] += 1
    else:
        dict_resTF[word] = 1
print dict_resTF


# 计算TF


wordLen = len(cutwords)
dict_TF = {}
for k, v in dict_resTF.iteritems():
    dict_TF[k] = v/wordLen
print dict_TF

# 返回格式  userId  word TF  word TF
