import argparse
import json
import jsonlines
import os, sys
sys.path.append(os.getcwd())
from preprocess.wikisql_preprocess import TextPreprocessor, drag_special_symbols
from tqdm import tqdm
import itertools
import numpy as np

def further_process(table_doc, retriever, processor):
    if retriever == "lexical":
        processed_table_doc = " , ".join(table_doc["headers"]) + " "
        for row in table_doc["rows"]:
            processed_table_doc += " , ".join(row)
            processed_table_doc += " "
        processed_table_doc = drag_special_symbols(processed_table_doc)
        processed_table_doc = processed_table_doc.lower()
        processed_table_doc = processor.preprocess(processed_table_doc, lowercase=True)
        return processed_table_doc
    else:
        return table_doc

def get_table_docs(tables, processor, args):
    """
        process tables that with database content (value)
    """
    tmp_tables = {}

    # to prevent duplicate tables
    for table in tables:
        try:
            tmp_tables[table['tableId']] = table
        except:
            print(table)
            raise
    
    print("distinct tables", len(tmp_tables.items()))

    processed_tables = {}
    too_long_table_num = 0
    seed_i = 0
    for table_id, table in tqdm(tmp_tables.items(), desc="Processing Tables"):
        new_table = {}
        headers = itertools.chain(*[column.values() for column in table["columns"]])
        headers = ["None" if header in [""," "] else header for header in headers]
        new_table["headers"] = headers if not args.append_title else [table["documentTitle"]] + headers

        all_rows = []
        for row in table["rows"]:
            new_row = itertools.chain(*[cell.values() for cell in row["cells"]])
            new_row = list(new_row)
            if args.max_cell_len > 0:
                new_row = [value[:args.max_cell_len] for value in new_row]
            all_rows.append(new_row)
        if len(all_rows) > args.max_row_num and args.max_row_num > 0:
            too_long_table_num += 1
            np.random.seed(seed_i)
            random_row_idx = np.random.permutation(range(len(all_rows)))[:args.max_row_num]
            all_rows = [all_rows[idx] for idx in random_row_idx]
        # rows are stored by row
        new_table["rows"] = all_rows
        
        # values are store by column
        values = []
        for column_idx in range(len(headers)):
            column_value = [row[column_idx] for row in all_rows]
            column_value = list(dict.fromkeys(column_value).keys())
            values.append(column_value)

        if args.retriever in ["dense"]:
            new_table["values"] = values if not args.append_title else [[]] + values

        new_table = further_process(new_table, args.retriever, processor)
        processed_tables[table_id] = new_table
        seed_i += 1

    print("too long table num: ", too_long_table_num)
    return processed_tables


def process_questions(questions_path, processor, retriever):
    questions, ground_truths, interaction_ids = [], [], []
    with jsonlines.open(questions_path) as reader:
        for obj in tqdm(reader, desc="Processing Questions"):
            assert len(obj["questions"]) == 1
            question = obj["questions"][0]["originalText"].lower()
            if retriever == "lexical":
                question = drag_special_symbols(question)
            questions.append(question)
            ground_truths.append([obj["table"]["tableId"]])
            interaction_ids.append(obj["questions"][0]["id"])
    if retriever == "lexical":
        questions = processor.preprocess(questions)
    return list(zip(questions, ground_truths, interaction_ids))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_table_corpus", action="store_true", default=False, help="whether to process table corpus")
    parser.add_argument("--process_questions", action="store_true", default=False, help="whether to process question")
    parser.add_argument("--tables_path", type=str, help="Path of the data file containing the tables to be retrieved.")
    parser.add_argument("--processed_tables_path", type=str, help="Path to output the processed tables")
    parser.add_argument("--questions_path", type=str, help="Path of data file containing questions")
    parser.add_argument("--output_dir", type=str, help="the output directory for saving processed tables and questions")
    parser.add_argument("--db_dir", type=str, help="Directory of sqlite3 databases")
    parser.add_argument("--retriever", type=str, choices=["dense"], help="whether the zero shot retriever is bert based")
    parser.add_argument("--append_title", action="store_true", default=False, help="whether to append table title as a special column")
    parser.add_argument("--max_row_num", type=int, help="max row number to sample")
    parser.add_argument("--max_cell_len", type=int, help="the maximum length of a table cell")
    parser.add_argument("--dataset", type=str, choices=["nqtable"], help="whether to process table qa of wikisql")

    args = parser.parse_args()

    processor = TextPreprocessor(spacy_model="en_core_web_sm")

    print("retriever:", args.retriever)

    if args.process_table_corpus:
        print("Processing table corpus...")
        with jsonlines.open(args.tables_path) as reader:
            tables = list(reader)
        print("number of tables:", len(tables))
        processed_tables = get_table_docs(tables, processor, args=args)

        if not os.path.exists(os.path.dirname(args.processed_tables_path)):
            os.mkdir(os.path.dirname(args.processed_tables_path))
        json.dump(processed_tables, open(args.processed_tables_path, "w"), ensure_ascii=False, indent=4)
    
    if args.process_questions:
        dataset_name = ""
        if "train" in args.questions_path:
            dataset_name = "train"
        elif "dev" in args.questions_path or "val" in args.questions_path:
            dataset_name = "dev"
        elif "test" in args.questions_path:
            dataset_name = "test"
        print("Processing question...")
        processed_questions = process_questions(args.questions_path, processor, args.retriever)

        json.dump(processed_questions, open(os.path.join(args.output_dir, "processed_questions_for_"+str(args.retriever).lower()+"_"+dataset_name+".json"), "w"), ensure_ascii=False, indent=4)
