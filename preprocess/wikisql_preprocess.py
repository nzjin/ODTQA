import argparse
from collections import defaultdict
import pickle
import json
import re
import spacy
import jsonlines
from os.path import abspath,join
import os, sys
import random
import numpy as np
sys.path.append(os.getcwd())


class TextPreprocessor():
    def __init__(self, spacy_model):
        self.nlp = spacy.load(spacy_model)

    def preprocess(self, txts, lowercase=True, remove_punct=True, 
                   remove_num=False, remove_stop=True, lemmatize=True):
        new_txts = []
        # print("Normalizing text...")
        for txt in txts:
            new_txts.append(self.preprocess_text(txt, lowercase, remove_punct, remove_num, remove_stop, lemmatize))
        return new_txts

    def preprocess_text(self, text, lowercase, remove_punct,
                        remove_num, remove_stop, lemmatize):
        if lowercase:
            text = self._lowercase(text)
        doc = self.nlp(text)
        if remove_punct:
            doc = self._remove_punctuation(doc)
        if remove_num:
            doc = self._remove_numbers(doc)
        if remove_stop:
            doc = self._remove_stop_words(doc)
        if lemmatize:
            text = self._lemmatize(doc)
        else:
            text = self._get_text(doc)
        return text

    def _lowercase(self, text):
        return text.lower()
    
    def _remove_punctuation(self, doc):
        return [t for t in doc if not t.is_punct]
    
    def _remove_numbers(self, doc):
        return [t for t in doc if not (t.is_digit or t.like_num or re.match('.*\d+', t.text))]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])

    def _get_text(self, doc):
        return ' '.join([t.text for t in doc])

def drag_special_symbols(text):
    text = re.sub("(\S)([\(\)\{\}\-\/])", r"\1 \2", text)
    text = re.sub("([\(\)\{\}\-\/])(\S)", r"\1 \2", text)
    return text

def further_process(table_doc, retriever, processor):
    if retriever == "lexical":
        processed_table_doc = " , ".join(table_doc["headers"]) + " "
        for row in table_doc["rows"]:
            processed_table_doc += " , ".join(row)
            processed_table_doc += " "
        processed_table_doc = drag_special_symbols(processed_table_doc)
        processed_table_doc = processor.preprocess(processed_table_doc)
        return processed_table_doc
    else:
        return table_doc

def get_table_docs(tables, processor, args):
    distinct_table_docs = []
    table_id_2_distinct_id = {}
    distinct_id_2_table_id = defaultdict(list)
    # agregate table by their headers, if the headers are same, then
    # two tables are the same
    for i, table in enumerate(tables):
        table_doc = ",".join(table["header"])
        try:
            distinct_id = distinct_table_docs.index(table_doc)
            table_id_2_distinct_id[table["id"]] = distinct_id
            distinct_id_2_table_id[str(distinct_id)].append((table["id"], i))
        except:
            # not in distinct table docs
            distinct_table_docs.append(table_doc)
            distinct_id = len(distinct_table_docs) - 1
            table_id_2_distinct_id[table["id"]] = distinct_id
            distinct_id_2_table_id[str(distinct_id)].append((table["id"], i))

    distinct_table_docs = defaultdict(dict)
    too_long_table_num = 0
    seed_i = 0
    for distinct_id, table_id_list in distinct_id_2_table_id.items():
        table_header = tables[table_id_list[0][1]]["header"]
        all_rows = []
        for table_id, idx in table_id_list:
            all_rows.extend(tables[idx]["rows"])
        if len(all_rows) > args.max_row_num:
            too_long_table_num += 1
            # make seed_i static, to ensure sampled rows are same for different models 
            np.random.seed(seed_i+666)
            # 666 means nothing here, treated as a initial seed
            random_row_idx = np.random.permutation(range(len(all_rows)))[:args.max_row_num]
            random_rows = [all_rows[idx] for idx in random_row_idx]
        else:
            random_rows = all_rows
        for i, row in enumerate(random_rows):
            # constrain the maximum length of a table cell
            if args.max_cell_len > 0:
                random_rows[i] = [str(ele)[:args.max_cell_len] for ele in row]
            else:
                random_rows[i] = [str(ele) for ele in row]
            # constrain the maximum length of a table cell
            
        table_docs = {
                "headers": table_header,
                "rows": random_rows}
        distinct_table_docs[int(distinct_id)] = further_process(table_docs, args.retriever, processor)
        seed_i += 1
    
    processed_tables = {
        "distinct_table_docs": distinct_table_docs,
        "table_id_2_distinct_id": table_id_2_distinct_id,
        "distinct_id_2_table_id": distinct_id_2_table_id,
        "too_long_table_num": too_long_table_num
    }
    print("too long table num: ", too_long_table_num)
    print("distinct table num: ", len(distinct_id_2_table_id.keys()))
    return processed_tables

def process_questions(questions_path, processed_tables, processor, retriever):
    questions, ground_truths, table_ids = [], [], []
    table_id_2_distinct_id = processed_tables["table_id_2_distinct_id"]
    with jsonlines.open(questions_path) as reader:
        for obj in reader:
            question = obj["question"].lower()
            if retriever == "lexical":
                question = drag_special_symbols(question)
            questions.append(question)
            ground_truths.append([table_id_2_distinct_id[obj["table_id"]]])
            table_ids.append(obj["table_id"])
    if retriever == "lexical":
        questions = processor.preprocess(questions, lemmatize=True)
    return list(zip(questions, ground_truths, table_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_table_corpus", action="store_true", default=False, help="whether to process table corpus")
    parser.add_argument("--process_questions", action="store_true", default=False, help="whether to process question")
    parser.add_argument("--tables_path", type=str, help="Path of the data file containing the tables to be retrieved.")
    parser.add_argument("--processed_tables_path", type=str, help="Path to output the processed tables")
    parser.add_argument("--questions_path", type=str, help="Path of data file containing questions")
    parser.add_argument("--output_dir", type=str, help="the output directory for saving processed tables and questions")
    parser.add_argument("--retriever", type=str, choices=["lexical","dense"], help="the type of retriever")
    parser.add_argument("--max_cell_len", type=int, default=-1, help="whether to strip the content of a cell under max cell len")
    parser.add_argument("--max_row_num", type=int, default=10, help="max row number to sample")

    args = parser.parse_args()

    processor = TextPreprocessor(spacy_model="en_core_web_sm")

    print("retriever:", args.retriever)

    if args.process_table_corpus:
        print("Processing table corpus...")
        tables = json.load(open(args.tables_path,"r"))

        processed_tables = get_table_docs(tables, processor, args=args)

        json.dump(processed_tables, open(args.processed_tables_path, "w"), ensure_ascii=False, indent=4)
    
    if args.process_questions:
        dataset_name = ""
        if "train" in args.questions_path:
            dataset_name = "train"
        elif "dev" in args.questions_path:
            dataset_name = "dev"
        elif "test" in args.questions_path:
            dataset_name = "test"
        print("Processing question...")
        if args.process_table_corpus:
            processed_tables = processed_tables
        else:
            processed_tables = json.load(open(args.processed_tables_path,"r"))
        processed_questions = process_questions(args.questions_path, processed_tables, processor, args.retriever)

        json.dump(processed_questions, open(join(args.output_dir, \
            "processed_questions_for_"+str(args.retriever).lower()+"_"+dataset_name+".json"), "w"), ensure_ascii=False, indent=4)
