import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re
import scipy.sparse as sp
from collections import defaultdict
import gc
from itertools import chain #用来将嵌套列表转为一维数组


def load_data():
    file_name = "E:\\data\\wanfang\\"
    final_name = "E:\\data\\wf_all\\"
    folder_list = os.listdir(file_name)
    for foler in tqdm(folder_list, leave=True, desc="read files"):
        try:
            final_name_txt = os.path.join(final_name, foler)
            final_name_txt = final_name_txt + '.txt'

            file_path = os.path.join(file_name, foler)
            files = os.listdir(file_path)
            f = open(final_name_txt, 'w', encoding='utf-8')
            for file_in in files:
                try:
                    file_p = file_path + '/' + file_in
                    for line in open(file_p, encoding='utf-8'):
                        f.writelines(line)
                except:
                    continue
            f.close()
        except:
            continue


def del_empty_file():
    file_name = "E:\\data\\wf_all\\"
    txt_file = os.listdir(file_name)
    # print(txt_file,len(txt_file))
    i = 0
    for txfile in tqdm(txt_file, leave=True, desc="del empty file..."):
        f_p = file_name + txfile
        if os.path.getsize(f_p) == 0:
            i += 1
            os.remove(f_p)
    print("totally delete empty ", i, "file..")


def test_read_txt():
    file_name = "E:\\data\\wf_all\\"
    txt_file = os.listdir(file_name)
    # for txfile in tqdm(txt_file, leave=True, desc="reading datas..."):
    f_name = "E:\\data\\wf_all\\zzqgyxy.txt"
    datas = np.loadtxt(f_name, encoding='utf-8', dtype=np.str, delimiter='\n')
    print(type(datas))

    for data in datas:
        data = str(data)
        try:
            paper_data = json.loads(data)
            key_word = paper_data['Keywords']
            label = paper_data['PeriodicalClassCode']
        except:
            continue
        if len(key_word) == 0 or len(label) == 0:
            continue
        print(key_word, "----", label)


def delInfo():
    file_name = "E:\\data\\wf_all\\"
    folder_list = os.listdir(file_name)
    d_name = "E:\\data\\wf_dic\\"
    for folder in tqdm(folder_list, leave=True, desc="process files..."):
        filename = os.path.join(file_name, folder)
        datas = np.loadtxt(filename, dtype=np.str, encoding='utf-8',
                           delimiter="\n")
        final_name = os.path.join(d_name, folder)
        f = open(final_name, 'w', encoding='utf-8')
        for data in datas:
            dic = {}
            data = str(data)
            try:
                data = data.replace('\'', '\"')
                data = json.loads(data)
                if ('PeriodicalClassCode' in data.keys() and (
                        'Keywords' in data.keys() or 'MachinedKeywords' in data.keys())):
                    code = data['PeriodicalClassCode']
                    if ('Keywords' in data.keys()):
                        words = data['Keywords']
                    else:
                        words = data["MachinedKeywords"]
                    dic['PeriodicalClassCode'] = code
                    dic['Keywords'] = words
                    strdic = str(dic) + '\n'
                    f.writelines(strdic)
            except:
                continue
        f.close()


def readfile():
    d_name = "E:\\data\\wf_dic\\21sjjzcl.txt"
    # with open(d_name,'r',encoding='utf-8') as f:
    #     line = f.readline()
    #     line = line.replace('\'',"\"")
    #     print(line,"==",type(line))
    #     dic = json.loads(line)
    #     print(dic,type(dic))
    datas = np.loadtxt(d_name, dtype=np.str, encoding='utf-8',
                       delimiter="\n")
    for data in datas:
        try:
            data = str(data)
            data = data.replace('\'', "\"")
            data = json.loads(data)
            keyword = data['Keywords']
            label = data['PeriodicalClassCode']
            print(keyword, '==', label)
        except:
            pass


def mergeFile():
    mergefile_one = "E:\\data\\wf_merge\\one.txt"
    mergefile_two = "E:\\data\\wf_merge\\two.txt"
    originfile = "E:\\data\\wf_dic\\"
    filenames = os.listdir(originfile)
    one_file = filenames[:2000]
    two_file = filenames[2000:]
    file_one = open(mergefile_one, 'w', encoding='utf-8')
    one = ["{}{}".format("E:\\data\\wf_dic\\", a) for a in one_file]
    two = ["{}{}".format("E:\\data\\wf_dic\\", a) for a in two_file]
    for file in tqdm(one, leave=True, desc="merge files one"):
        for line in open(file, encoding='utf-8'):
            file_one.writelines(line)
    file_one.close()
    file_two = open(mergefile_two, 'w', encoding='utf-8')
    for file in tqdm(two, leave=True, desc="merge files one"):
        for line in open(file, encoding='utf-8'):
            file_two.writelines(line)
    file_two.close()


def readline():
    mergefile_one = "E:\\data\\wf_merge\\one.txt"
    i = 0
    for line in tqdm(open(mergefile_one, encoding='utf-8'), leave=False, desc="read"):
        try:
            data = str(line)
            data = data.replace('\'', '\"')
            data = json.loads(data)
            print(data['Keywords'], '===', data['PeriodicalClassCode'])
            i += 1
        except:
            continue
    print(i)


def getline():
    labels = []
    key_words = []
    unique_key_word = []
    unique_label = []
    mergefile_one = "E:\\data\\wf_merge\\two.txt"
    fname = "E:\\data\\wf_merge\\3.txt"
    # f= open(fname,'w',encoding='utf-8')
    for line in tqdm(open(mergefile_one, encoding='utf-8'), leave=True, desc="choosing.."):
        # f.writelines(line)
        try:
            data = str(line)
            data = data.replace('\'', '\"')
            paper_data = json.loads(data)
            key_word = paper_data['Keywords']
            label = paper_data['PeriodicalClassCode']
        # f.close()
        except:
            continue
        labels.append(label)
        key_words.append(key_word)
        # unique_key_word = unique_key_word + key_word
        for key in key_word:
            unique_key_word.append(key)


def readtxt():
    file = '../uni.txt'
    arr = np.unique(np.array(["好的", "不知道"]))
    pickle.dump(arr, open(file, 'wb'))  # 存储numpy格式数据
    print("====")
    # with open(file, 'rb') as f2:
    #     b = pickle.load(f2)#读取numpy格式诗句
    #     print(b)
    # with open('../unique_key_word1.txt', 'rb') as f:
    #     b = pickle.load(f)
    #     # b = list(b)
    #     print(b)
    with open('../unique_key_word2.txt', 'rb') as ff:
        c = pickle.load(ff)
        c = list(c)
    with open('../unique_key_word1.txt', 'rb') as f1:
        u1 = pickle.load(f1)
    with open('./unique_key_word2.txt', 'rb') as f2:
        u2 = pickle.load(f2)
    with open('./unique_key_word3.txt', 'rb') as f3:
        u3 = pickle.load(f3)
    with open('./unique_key_word4.txt', 'rb') as f4:
        u4 = pickle.load(f4)


def testSparse():
    feat = sp.csc_matrix((10, 20))
    kd = defaultdict(int)
    kd['你好'] = 1
    kd['背景'] = 2
    kd['广州'] = 3
    key_words = [['你好', '广州', '背景']]
    for i, keys in enumerate(key_words):
        # print(type([kd[kes] for kes in keys]))
        # feat[i,[kd[kes] for kes in keys]] = 1
        lp = []
        for kes in keys:
            lp.append(kd[kes])
        print(lp, "here")
        print(type([kd[kes] for kes in keys]), [kd[kes] for kes in keys])
        # feat[i, [kd[kes] for kes in keys]] = 1
        feat[i, lp] = 1
        print("1111")


def testEnumber():
    # with open('./unique_key_word1.txt', 'rb') as f1:
    #     u1 = pickle.load(f1)
    with open('../unique_key_word2.txt', 'rb') as f2:
        u2 = pickle.load(f2)
    # with open('../unique_key_word3.txt', 'rb') as f3:
    #     u3 = pickle.load(f3)
    # with open('./unique_key_word4.txt', 'rb') as f4:
    #     u4 = pickle.load(f4)
    # unique_key_word = np.unique(np.append(u2, u3))
    unique_key_word = u2
    with open('../key_words.txt', 'rb') as t3:
        key_words = pickle.load(t3)
    num_keywords = len(unique_key_word)
    # feat1 = sp.csc_matrix((len(key_words), num_keywords))
    feat = sp.coo_matrix((len(key_words), num_keywords), dtype=np.int64)
    print(len(key_words))
    keyword_dict = defaultdict(int)
    for i, key in tqdm(enumerate(unique_key_word), desc="keydicting...."):
        keyword_dict[key] = i
    del unique_key_word, u2
    gc.collect()
    tcol = []
    tdata = []
    trow = []
    print(len(key_words))
    j = 0
    for i, keys in tqdm(enumerate(key_words), leave=True, desc="reflexing1.."):
        j += 1
        if (j > 100000):
            break
        # feat[i, [keyword_dict[kes] for kes in keys]] = 1
        ttemp = []
        for kes in keys:
            temp = keyword_dict[kes]
            ttemp.append(temp)
        tcol.append(ttemp)  # 列添加
    na = list(chain(*tcol))
    feat.col = np.array(na)
    del tcol, keyword_dict, f2, na
    gc.collect()
    j = 0
    for i, keys in tqdm(enumerate(key_words), leave=True, desc="reflexing2.."):
        j += 1
        if (j > 100000):
            break
        num = len(keys)
        a = np.ones(num) * i
        trow.append(a)  # 行添加
    na = list(chain(*trow))
    feat.row = np.array(na)
    del trow, na
    gc.collect()
    j = 0
    for i, keys in tqdm(enumerate(key_words), leave=True, desc="reflexing3.."):
        j += 1
        if (j > 100000):
            break
        num = len(keys)
        tdata.append((list(np.ones(num))))  # 数据添加
    na = list(chain(*tdata))
    feat.data = np.array(na)
    del tdata, key_words, na
    gc.collect()
    print("over")
    feat2 = feat.astype(dtype=np.int64).tocsc()
    print("transover")

def readmodel():
    pass

if __name__ == '__main__':
    # load_data() #处理文件夹，生成txt
    # del_empty_file() #把空txt处理掉
    # test_read_txt()
    # delInfo()
    # readfile()
    # mergeFile()
    # readline()
    # getline()
    # readtxt()
    # testSparse()
    testEnumber()
