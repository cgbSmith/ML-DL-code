import json
import pickle
import numpy as np
import re
import json
import time
from collections import defaultdict

if __name__ == '__main__':
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
    test_data = np.loadtxt("2.txt", encoding='utf-8', dtype=np.str, delimiter="\n")
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

    # start = time.time()
    # model_1 = pickle.load(open("model.dat", "rb"))
    #
    # num_keywords = np.load('keywords.npy')
    # print(num_keywords)
    #
    # with open('kd.txt', "r" , encoding='utf-8') as f:
    #     string_1 = f.read()
    #     # print(type(string_1))
    #
    # data = string_1[27:-1]
    # keyword_dict = eval(data)
    #
    # str1 = np.loadtxt("2.txt", encoding='utf-8', dtype=np.str, delimiter="\n")
    # str1 = str(str1)
    # str1 = json.loads(str1)
    # print(str1, "---", type(str1))
    # kw = str1['Keywords']
    # # print(type(kw))
    # la = str1['PeriodicalClassCode']
    # feat1 = np.zeros(num_keywords, dtype=np.int32)
    # for kes in kw:
    #     if kes in keyword_dict.keys():
    #         feat1[keyword_dict[kes]] = 1
    # feat1 = feat1.reshape(1, -1)
    # print(model_1.predict(feat1))
    # end = time.time()
    # print("runtime = ", end - start)


