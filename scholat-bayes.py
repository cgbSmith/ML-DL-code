import json
import os
import random
from sklearn.naive_bayes import MultinomialNB
import time
from ctypes import Array
from tqdm import tqdm


def TextProcess(folder_path, test_size):
    folder_list = os.listdir(folder_path)
    # print(folder_list)
    data_list = []
    class_list = []
    i = 1
    num = 0
    # for folder in folder_list:
    for t in tqdm((folder_list)):
        print("t = ",t)
        folder = t
        if i > 10:
            break
        i += 1
        new_folder_path = os.path.join(folder_path, folder)
        # print("new_folder_path", new_folder_path)
        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            # print("----",file)
            if j > 110:
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                # path=os.path.join(new_folder_path,file)
                content = f.read()
                raw = content.split('\n')
                raw = raw[:-1]
                for item in raw:
                    item = json.loads(item)
                    # print(item.get("OriginalClassCode"))
                    # print(item.get("Keywords"))
                    # classCode = item.get("OriginalClassCode")
                    classCode = item.get("PeriodicalClassCode")
                    Title = item.get("Title")
                    # classCode = item.get("MachinedClassCode")
                    keyWords = item.get("Keywords")

                    if classCode != None and keyWords != None:
                        data_list.append(keyWords)
                        class_list.append(classCode[0])
                        # print(classCode[0], "-----", keyWords, "==============", Title)
                        num += 1

    data_class_list = list(zip(data_list, class_list))  # zip压缩，将数据和标签对应压缩
    # print(data_class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)
    # print("train",train_data_list,"----",train_class_list)
    # print("test",test_data_list,"===",test_class_list)
    all_word_dict = {}  # 统计训练集的词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_word_dict.keys():
                all_word_dict[word] += 1
            else:
                all_word_dict[word] = 1

    # print("all_word_dic", all_word_dict)
    # 根据键值排序
    all_word_dict_tuple = sorted(all_word_dict.items(), key=lambda f: f[1], reverse=True)
    # print(all_word_dict_tuple)

    all_words_list, all_word_nums = zip(*all_word_dict_tuple)

    all_words_list = list(all_words_list)
    # print(train_class_list,"\n\n\n",test_class_list)
    # print(all_words_list)
    print("共有数据", num, "条")
    # print(train_data_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list):
    feature_words = []
    for t in range(len(all_words_list)):
        if not all_words_list[t].isdigit() and 1 < len(all_words_list[t]) < 10:
            feature_words.append(all_words_list[t])
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def ArticleClassfier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    start_time = time.time()
    folder_path = 'E:\data\wanfang'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcess(folder_path, 0.1)
    feature_words = words_dict(all_words_list)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_acc = ArticleClassfier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    end_time = time.time()
    run_time = end_time - start_time
    print("acc:=", test_acc)
    print("runtime:=", run_time)
