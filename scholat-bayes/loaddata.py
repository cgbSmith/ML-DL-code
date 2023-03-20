import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re


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
    file_one = open(mergefile_one,'w',encoding='utf-8')
    one = ["{}{}".format("E:\\data\\wf_dic\\",a) for a in one_file]
    two = ["{}{}".format("E:\\data\\wf_dic\\",a) for a in two_file]
    for file in tqdm(one,leave=True,desc="merge files one"):
        for line in open(file,encoding='utf-8'):
            file_one.writelines(line)
    file_one.close()
    file_two = open(mergefile_two,'w',encoding='utf-8')
    for file in tqdm(two,leave=True,desc="merge files one"):
        for line in open(file,encoding='utf-8'):
            file_two.writelines(line)
    file_two.close()



def readline():
    mergefile_one = "E:\\data\\wf_merge\\one.txt"
    i =0
    for line in tqdm(open(mergefile_one,encoding='utf-8'),leave=False,desc="read"):
        try:
            data = str(line)
            data  =  data.replace('\'','\"')
            data = json.loads(data)
            print(data['Keywords'],'===',data['PeriodicalClassCode'])
            i+=1
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
    for line in tqdm(open(mergefile_one,encoding='utf-8'),leave=True,desc="choosing.."):
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


if __name__ == '__main__':
    # load_data() #处理文件夹，生成txt
    # del_empty_file() #把空txt处理掉
    # test_read_txt()
    # delInfo()
    # readfile()
    # mergeFile()
    # readline()
    getline()