import gc

import numpy as np
import os
import json
from collections import defaultdict
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB
import pickle
import time
import torch
from tqdm import tqdm
import gc
from itertools import chain
from sklearn.externals import joblib


def read_datasets(root="E:\\data\\wf_merge\\"):
    folder_list = os.listdir(root)
    labels = []
    key_words = []
    unique_key_word = []
    unique_label = []
    for folder in folder_list:  # 获得所有文件夹
        folder_list = ["{}{}".format("E:\\data\\wf_merge\\", a) for a in folder_list]
    # need
    for folder in tqdm((folder_list), leave=True, desc="read files"):
        # for folder in folder_list:
        try:
            print(folder)
            # datas = np.loadtxt(folder, dtype=np.str, encoding='utf-8',
            #                    delimiter="\n")
            # for data in tqdm(datas, leave=False, desc="read line..."):
            for line in tqdm(open(folder, encoding='utf-8'), leave=True, desc="readline..."):
                try:
                    data = str(line)
                    data = data.replace('\'', '\"')
                    paper_data = json.loads(data)
                    key_word = paper_data['Keywords']
                    label = paper_data['PeriodicalClassCode']
                except:
                    continue
                labels.append(label)
                key_words.append(key_word)
                ## --------------
                ## unique_key_word = unique_key_word + key_word
                ## unique_label = unique_label + label
                ## -------------
                for key in key_word:
                    unique_key_word.append(key)
                for l in label:
                    unique_label.append(l)
        except:
            continue
    # need
    # pickle.dump(unique_label, open("unique_label.txt", 'wb'))
    # pickle.dump(labels, open("labels.txt", 'wb'))
    # pickle.dump(key_words, open('key_words.txt', 'wb'))
    # with open('./unique_label.txt', 'rb') as t1:
    #     unique_label = pickle.load(t1)
    # with open('./labels.txt', 'rb') as t2:
    #     labels = pickle.load(t2)
    # with open('./key_words.txt', 'rb') as t3:
    #     key_words = pickle.load(t3)
    # del t2, t3
    gc.collect()
    one_list = unique_key_word[0:20000000]
    two_list = unique_key_word[20000000:30000000]
    three_list = unique_key_word[30000000:40000000]
    four_list = unique_key_word[40000000:50000000]
    five_list = unique_key_word[50000000:65000000]
    six_list = unique_key_word[65000000:75000000]
    seven_list = unique_key_word[75000000:85000000]
    eight_list = unique_key_word[85000000:-1]
    one_list = np.unique(np.array(one_list))
    two_list = np.unique(np.array(two_list))
    three_list = np.unique(np.array(three_list))
    four_list = np.unique(np.array(four_list))
    five_list = np.unique(np.array(five_list))
    six_list = np.unique(np.array(six_list))
    seven_list = np.unique(np.array(seven_list))
    eight_list = np.unique(np.array(eight_list))
    del unique_key_word, one_list, two_list, three_list, four_list, five_list, six_list, seven_list, eight_list
    gc.collect()
    # unique_key_word = np.unique(np.array(unique_key_word))
    u1 = np.unique(np.append(one_list, two_list))
    u2 = np.unique(np.append(three_list, four_list))
    u3 = np.unique(np.append(five_list, six_list))
    u4 = np.unique(np.append(seven_list, eight_list))
    uu1 = np.unique(np.append(u1, u2))
    uu2 = np.unique(np.append(u3, u4))
    del u1, u2, u3, u4
    gc.collect()
    unique_key_word = np.unique(np.append(uu1, uu2))
    # pickle.dump(u1, open("unique_key_word1.txt", 'wb'))
    # pickle.dump(u2, open("unique_key_word2.txt", 'wb'))
    # pickle.dump(u3, open("unique_key_word3.txt", 'wb'))
    # pickle.dump(u4, open("unique_key_word4.txt", 'wb'))
    # with open('./unique_key_word1.txt', 'rb') as f1:
    #     u1 = pickle.load(f1)
    # with open('./unique_key_word2.txt', 'rb') as f2:
    #     u2 = pickle.load(f2)
    # uu1 = np.unique(np.append(u1, u2))
    # del u1, u2, f1, f2
    # gc.collect()
    # with open('./unique_key_word3.txt', 'rb') as f3:
    #     u3 = pickle.load(f3)
    # with open('./unique_key_word4.txt', 'rb') as f4:
    #     u4 = pickle.load(f4)
    # uu2 = np.unique(np.append(uu1, u3))
    # unique_key_word = np.unique(np.append(uu2, u4))
    # del u3, u4, uu1, uu2, f3, f4
    # gc.collect()
    # unique_key_word = np.unique(np.append(uu1, uu2))
    print("over")
    # pickle.dump(unique_key_word, open("unique_key_word.txt", 'wb'))
    # print(unique_key_word)
    # 构建字典
    keyword_dict = defaultdict(int)
    for i, key in enumerate(unique_key_word):
        keyword_dict[key] = i
    print("keyword_dict ok")
    num_keywords = len(unique_key_word)
    key_words = np.array(key_words)
    print("keyword ok")
    # feat = sp.csc_matrix((len(key_words), num_keywords))
    feat = sp.coo_matrix((len(key_words), num_keywords), dtype=np.int64)
    print("len keywords == ", len(key_words))
    print("feat ok")
    # feat = np.zeros(shape=(len(key_words), num_keywords))
    for i, key in tqdm(enumerate(unique_key_word), desc="keydicting...."):
        keyword_dict[key] = i
    del unique_key_word
    gc.collect()
    tcol = []
    tdata = []
    trow = []
    print(len(key_words))
    for i, keys in tqdm(enumerate(key_words), leave=True, desc="reflexing1.."):
        # feat[i, [keyword_dict[kes] for kes in keys]] = 1
        ttemp = []
        for kes in keys:
            temp = keyword_dict[kes]
            ttemp.append(temp)
        tcol.append(ttemp)  # 列添加
    na = list(chain(*tcol))
    feat.col = np.array(na)
    # 存储keydict
    kdict = str(keyword_dict)
    f = open("keyword_dict.txt", "w", encoding='utf-8')
    f.write(kdict)
    f.close()
    print("keydict存储完毕")
    del tcol, keyword_dict, na
    gc.collect()
    for i, keys in tqdm(enumerate(key_words), leave=True, desc="reflexing2.."):
        num = len(keys)
        a = np.ones(num) * i
        trow.append(a)  # 行添加
    na = list(chain(*trow))
    feat.row = np.array(na)
    del trow, na
    gc.collect()
    for i, keys in tqdm(enumerate(key_words), leave=True, desc="reflexing3.."):
        num = len(keys)
        tdata.append((list(np.ones(num))))  # 数据添加
    na = list(chain(*tdata))
    feat.data = np.array(na)
    del tdata, key_words, na
    gc.collect()
    print("over")
    feat_end = feat.astype(dtype=np.int64).tocsc()
    print("last ok")
    # feat[i][[keyword_dict[kes] for kes in keys]] = 1
    # return feat, keyword_dict, num_keywords, np.array(labels), key_words
    return feat_end, num_keywords, np.array(labels)


def ArticleClassfier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    print('trainning......')
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    print('trainning over')
    try:
        # pickle.dump(classifier, open("model_all.dat", "wb"), protocol=4)  # 保存模型
        test_accuracy = classifier.score(test_feature_list, test_class_list)
        # joblib.dump(classifier,'bayesmodel.pkl')
    except:
        print('保存失败')
    return test_accuracy
    # 载入模型可以用 joblib.load('bayesmodel.pkl')


def get_mask(y, train_ratio, test_ratio, device=None):
    if device is None:
        device = torch.device("cpu")
    train_indexes = list()
    test_indexes = list()
    val_indexes = list()
    # npy = np.array(list(chain(*list(y))))
    npy = y

    def get_sub_mask(sub_x_indexes):
        np.random.shuffle(sub_x_indexes)
        sub_train_count = int(len(sub_x_indexes) * train_ratio)
        # sub_test_count = int(len(sub_x_indexes) * test_ratio)
        sub_train_indexes = sub_x_indexes[0:sub_train_count]
        sub_test_indexes = sub_x_indexes[sub_train_count:]
        # sub_val_indexes = sub_x_indexes[sub_train_count + sub_test_count:]
        return sub_train_indexes, sub_test_indexes

    def flatten_np_list(np_list):
        total_size = sum([len(item) for item in np_list])
        result = np.ndarray(shape=total_size)
        last_i = 0
        for item in np_list:
            result[last_i:last_i + len(item)] = item
            last_i += len(item)
        return np.sort(result)

    # np.unique: 去除重复的数值
    for class_id in np.unique(npy):
        indexes = np.argwhere(npy == class_id).flatten().astype(int)  # 获取ID为对应label的节点下标
        m, n = get_sub_mask(indexes)
        train_indexes.append(m)
        test_indexes.append(n)
        # val_indexes.append(q)
    train_indexes = torch.LongTensor(flatten_np_list(train_indexes)).to(device)
    test_indexes = torch.LongTensor(flatten_np_list(test_indexes)).to(device)
    # val_indexes = torch.LongTensor(flatten_np_list(val_indexes)).to(device)
    return train_indexes.numpy(), test_indexes.numpy()


if __name__ == "__main__":
    start = time.time()
    # feat, keyword_dict, num_keywords, labels, keywords = read_datasets()
    # feat, num_keywords, labels = read_datasets()
    # np.save("num_key_words", num_keywords)
    # kdict = str(keyword_dict)
    # f = open("keyword_dict.txt", "w", encoding='utf-8')
    # f.write(kdict)
    # f.close()
    # train_indexes, test_indexes, val_indexes = get_mask(labels, 0.8, 0.1)
    # test_accuracy = ArticleClassfier(feat[train_indexes], feat[test_indexes], labels[train_indexes],labels[test_indexes])
    testrate = 0.3
    trainrate = 0.6
    labels = np.array([['TU'], ['TA'], ['UA'], ['UA'], ['TA'],['TA']])
    for i in range(9):
        train_indexes, test_indexes = get_mask(labels, trainrate, testrate)
        trainrate += 0.03
        testrate -= 0.03
        test_accuracy = ArticleClassfier(feat[train_indexes], feat[test_indexes], labels[train_indexes],
                                         labels[test_indexes])
        print("=============================")
        print("trainrate = ", trainrate, "  testrate = ", testrate, " acc = ", test_accuracy)
        end = time.time()
        print("run_time = ", end - start)
        print("=============================")
