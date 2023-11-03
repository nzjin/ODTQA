import jsonlines
import argparse
import json
import os, sys
parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str)
parser.add_argument("--dev_path", type=str)
parser.add_argument("--test_path", type=str)
parser.add_argument("--partial_tables_path", type=str, help="Path to output tables used in train, dev, test path")

args = parser.parse_args()


with jsonlines.open(args.train_path) as reader:
    train_set = list(reader)

with jsonlines.open(args.dev_path) as reader:
    dev_set = list(reader)

with jsonlines.open(args.test_path) as reader:
    test_set = list(reader)

dataset =  train_set + dev_set + test_set

if not os.path.exists(os.path.dirname(args.partial_tables_path)):
    os.mkdir(os.path.dirname(args.partial_tables_path))
with jsonlines.open(args.partial_tables_path, mode='w') as writer:
    for sample in dataset:
        writer.write(sample['table'])



