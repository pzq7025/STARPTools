# # 转换olid数据为lyrics的风格
# python style_data_transfer.py --model_dir ./model/lyrics --top_p_value 0.1 --datasets olid

# 转换wiki数据集，使用lyrics的风格
python style_data_transfer.py --model_dir ./model/lyrics --top_p_value 0.7 --file_path ./bad_bert_data/wiki-clean/train.txt --file_type text --data_type wiki