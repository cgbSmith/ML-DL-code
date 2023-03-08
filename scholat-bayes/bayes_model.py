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


def read_datasets(root="E:\\data\\wanfang\\"):
    folder_list = os.listdir(root)
    labels = []
    key_words = []
    unique_key_word = []
    unique_label = []
    # folder_list = folder_list[:10]
    i = 0
    for folder in folder_list:  # 获得所有文件夹
        print(i, folder)
        i += 1
        files = os.listdir(root + folder)  # 获取所有文件
        adds = os.path.join(root, folder)
        for file in files:
            try:
                file_adds = os.path.join(adds, file)
                # print(file_adds)
                datas = np.loadtxt(file_adds, dtype=np.str, encoding='utf-8',
                                   delimiter="\n")
                # print(datas)
                for data in datas:
                    data = str(data)
                    paper_data = json.loads(data)
                    try:
                        key_word = paper_data['Keywords']
                        label = paper_data['PeriodicalClassCode']
                    except:
                        continue
                    if len(key_word) == 0 or len(label) == 0:
                        continue
                    labels.append(label)
                    key_words.append(key_word)
                    unique_key_word = unique_key_word + key_word
                    unique_label = unique_label + label

            except:
                continue
    unique_key_word = np.unique(np.array(unique_key_word))
    # print(unique_key_word)
    # 构建字典
    keyword_dict = defaultdict(int)
    for i, key in enumerate(unique_key_word):
        keyword_dict[key] = i
    num_keywords = len(unique_key_word)
    key_words = np.array(key_words)
    feat = sp.csc_matrix((len(key_words), num_keywords))
    # feat = np.zeros(shape=(len(key_words), num_keywords))
    for i, keys in enumerate(key_words):
        feat[i, [keyword_dict[kes] for kes in keys]] = 1
        # feat[i][[keyword_dict[kes] for kes in keys]] = 1
    return feat, keyword_dict, num_keywords, np.array(labels), key_words


def ArticleClassfier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    pickle.dump(classifier, open("model_all.dat", "wb"))  # 保存模型
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


def get_mask(y, train_ratio=0.6, test_ratio=0.2, device=None):
    if device is None:
        device = torch.device("cpu")
    train_indexes = list()
    test_indexes = list()
    val_indexes = list()
    npy = y

    def get_sub_mask(sub_x_indexes):
        np.random.shuffle(sub_x_indexes)
        sub_train_count = int(len(sub_x_indexes) * train_ratio)
        sub_test_count = int(len(sub_x_indexes) * test_ratio)
        sub_train_indexes = sub_x_indexes[0:sub_train_count]
        sub_test_indexes = sub_x_indexes[sub_train_count:sub_train_count + sub_test_count]
        sub_val_indexes = sub_x_indexes[sub_train_count + sub_test_count:]
        return sub_train_indexes, sub_test_indexes, sub_val_indexes

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
        m, n, q = get_sub_mask(indexes)
        train_indexes.append(m)
        test_indexes.append(n)
        val_indexes.append(q)
    train_indexes = torch.LongTensor(flatten_np_list(train_indexes)).to(device)
    test_indexes = torch.LongTensor(flatten_np_list(test_indexes)).to(device)
    val_indexes = torch.LongTensor(flatten_np_list(val_indexes)).to(device)
    return train_indexes.numpy(), test_indexes.numpy(), val_indexes.numpy()


if __name__ == "__main__":
    start = time.time()
    feat, keyword_dict, num_keywords, labels, keywords = read_datasets()
    np.save("Num_key_words", num_keywords)
    print(type(keyword_dict))
    kdict = str(keyword_dict)
    f = open("keyword_dict.txt", "w", encoding='utf-8')
    f.write(kdict)
    f.close()
    train_indexes, test_indexes, val_indexes = get_mask(labels, 0.8, 0.1)
    test_accuracy = ArticleClassfier(feat[train_indexes], feat[test_indexes], labels[train_indexes],
                                     labels[test_indexes])
    print(test_accuracy)
    end = time.time()
    print("run_time = ", end - start)

'''
    # 加载模型，然后进行数据测试
    start = time.time()
    model_1 = pickle.load(open("model2.dat", "rb"))
    # 读取num_keywords
    num_keywords = np.load('Num_key_words.npy')
    # 读取key_dict，然后继续处理
    with open('keyword_dict.txt', "r", encoding='utf-8') as f:
        ky_dict = f.read()
        # print(type(string_1))
    data = ky_dict[27:-1]
    keyword_dict = eval(data)
    # ================
    # 测试
    test_data = np.loadtxt("1.txt", encoding='utf-8', dtype=np.str, delimiter="\n")
    test_data = str(test_data)
    test_data = json.loads(test_data)
    kw = test_data['Keywords']
    la = test_data['PeriodicalClassCode']
    feat1 = np.zeros(num_keywords, dtype=np.int32)
    for kes in kw:
        if kes in keyword_dict.keys():
            feat1[keyword_dict[kes]] = 1
    feat1 = feat1.reshape(1, -1)
    end = time.time()
    print(model_1.predict(feat1))
    print("run_time = ", end - start) 
'''
