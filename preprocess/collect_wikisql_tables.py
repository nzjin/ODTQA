import jsonlines
import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("--train_tables_path", type=str)
parser.add_argument("--dev_tables_path", type=str)
parser.add_argument("--test_tables_path", type=str)
parser.add_argument("--tables_path", type=str, help="Path to output the merged tables")

args = parser.parse_args()

train_tables = jsonlines.Reader(open(args.train_tables_path,"r"))
dev_tables = jsonlines.Reader(open(args.dev_tables_path,"r"))
test_tables = jsonlines.Reader(open(args.test_tables_path,"r"))

tables = list(train_tables) + list(dev_tables) + list(test_tables)

if not os.path.exists(os.path.dirname(args.tables_path)):
    os.mkdir(os.path.dirname(args.tables_path))
json.dump(tables, open(args.tables_path, "w"), indent=4, ensure_ascii=False)