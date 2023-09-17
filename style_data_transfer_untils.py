import pandas as pd
import os

def n_test_sample(datasets:str, sample_size:int):
    """
    从测试集中抽样训练集数据
    """
    data_path = f"./original_model/datasets/{datasets}/{datasets}_test.csv"
    pf = pd.read_csv(data_path)
    print(f"The total number of data:{len(pf)}")
    pf = pf[pf['text'].str.len() >20]
    print(f"the length of data after detail:{len(pf)}")
    sample_pf = pf.sample(sample_size, random_state=76)
    print(f"The total number of sampled data:{len(sample_pf)}")
    return sample_pf["text"].to_list()


def write_test_file(convert_data, original_data, datasets:str, style_format, p_value):
    dir = "./detectGPT/datasets"
    # if not os.path.exists(dir):
    #     print(f"The path:{dir} not find and already creat the path!")
    #     os.makedirs(dir)
    p_value = str(p_value).replace(".", "_")
    data_path = f"{dir}/{datasets}_{style_format}_{p_value}_test.csv"
    labels = [1] * len(convert_data) + [0] * len(original_data)
    total_data = pd.DataFrame({"text": convert_data + original_data, "label": labels}, columns=["text", "label"])
    total_data.to_csv(data_path, index=False)
    print(f"Storing path:{data_path}")
    print(f"checking file name:{datasets}_{style_format}_{p_value}_test")


def n_test_sample_from_text(file_path:str, sample_size:int):
    """
    从txt文件中读取数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = [i.strip() for i in f.readlines()[:sample_size]]
    return content

def write_test_text_file(convert_data, original_data, datasets:str, style_format, p_value, data_type):
    dir = "./bad_bert_data/convert_data"
    # if not os.path.exists(dir):
    #     print(f"The path:{dir} not find and already creat the path!")
    #     os.makedirs(dir)
    p_value = str(p_value).replace(".", "_")
    data_path = f"{dir}/{data_type}_{datasets}_{style_format}_{p_value}_test.csv"
    labels = [1] * len(convert_data) + [0] * len(original_data)
    total_data = pd.DataFrame({"text": convert_data + original_data, "label": labels}, columns=["text", "label"])
    total_data.to_csv(data_path, index=False)
    print(f"Storing path:{data_path}")
    print(f"checking file name:{datasets}_{style_format}_{p_value}_test")