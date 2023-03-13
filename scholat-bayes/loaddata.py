import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json


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


if __name__ == '__main__':
    # load_data() #处理文件夹，生成txt
    # del_empty_file() #把空txt处理掉
    test_read_txt()
