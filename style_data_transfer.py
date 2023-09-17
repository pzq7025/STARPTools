import argparse
import logging
import sys
import torch
from tqdm import tqdm

from style_paraphrase.inference_utils import GPT2Generator
from style_data_transfer_untils import n_test_sample, write_test_file, n_test_sample_from_text, write_test_text_file

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="paraphraser_gpt2_large", type=str)
parser.add_argument('--top_p_value', default=0.7, type=float)
parser.add_argument('--datasets', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--sample_size', default=500, type=int)
parser.add_argument('--file_type', default="csv", type=str)
parser.add_argument('--file_path', default="text", type=str)
parser.add_argument('--data_type', default="wiki", type=str)
args = parser.parse_args()

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()


def transfer_data():
    print("Start sampling data...")
    if args.file_type == "csv":
        original_data = n_test_sample(datasets=args.datasets, sample_size=args.sample_size)
    if args.file_type == "text":
        original_data = n_test_sample_from_text(args.file_path, sample_size=args.sample_size)
    print("Start style transfer...")
    outputs = []
    for i in tqdm(range(0, len(original_data), args.batch_size), desc="minibatches done..."):
        generations, _ = paraphraser.generate_batch(original_data[i:i + args.batch_size])
        outputs.extend(generations)

    print("Start writing to file...")
    if args.file_type == "csv":
        write_test_file(outputs, original_data, datasets=args.datasets, style_format=style_format, p_value=args.top_p_value)
    if args.file_type == "text":
        write_test_text_file(outputs, original_data, datasets=args.datasets, style_format=style_format, p_value=args.top_p_value, data_type=args.data_type)
    print("Finished!")


if __name__ == "__main__":
    style_format = args.model_dir.split("/")[-1]
    print(f"Loading paraphraser..., type:{style_format}")
    paraphraser = GPT2Generator(args.model_dir, upper_length="same_5")
    paraphraser.modify_p(top_p=args.top_p_value)
    print(f"top_p_value:{args.top_p_value}")
    transfer_data()
    
