import json
import os
import random
from ctypes import Array


def TextProcess(folder_path, test_size):
    folder_list = os.listdir(folder_path)
    # print(folder_list)
    data_list = []
    class_list = []
    i = 1
    num = 0
    for folder in folder_list:
        if i > 10:
            break
        i += 1
        new_folder_path = os.path.join(folder_path, folder)
        # print("new_folder_path", new_folder_path)
        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            # print("----",file)
            if j > 15:
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
                    classCode = item.get("OriginalClassCode")
                    keyWords = item.get("Keywords")
                    if classCode != None and keyWords != None:
                        data_list.append(keyWords)
                        class_list.append(classCode[0])
                        # print(classCode,"-----",keyWords)
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
    print(all_word_dict_tuple)

    all_words_list, all_word_nums = zip(*all_word_dict_tuple)

    all_words_list = list(all_words_list)

    print("\n\n\n",all_words_list)
    print("共有数据", num, "条")
    # print(data_list)
    # print(class_list)


if __name__ == '__main__':
    folder_path = 'E:\data\wanfang'
    TextProcess(folder_path, 0.6)
    # "=============="
    # new_path = "E:\data\wanfang"
    # files = os.listdir(new_path)
    # print(type(files[0]))
    # new_path = os.path.join(new_path, files[0])
    # print(new_path)
    # files = os.listdir(new_path)
    # # print("-",files)
    # new_path = os.path.join(new_path, files[0])
    # print(new_path)
    # with open(new_path, 'r', encoding='utf-8') as f:
    #    content=f.read()
    #    # print(type(content),content)
