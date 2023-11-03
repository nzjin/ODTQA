# dataset="wikisql"
train_tables_path="data/wikisql/train.tables.jsonl"
dev_tables_path="data/wikisql/dev.tables.jsonl"
test_tables_path="data/wikisql/test.tables.jsonl"
tables_path="data/wikisql/processed/all.tables.json"
output_dir="data/wikisql/processed/"

# collect all tables
python preprocess/collect_wikisql_tables.py --train_tables_path $train_tables_path --dev_tables_path $dev_tables_path --test_tables_path $test_tables_path --tables_path $tables_path


train_questions_path="data/wikisql/train.jsonl"
dev_questions_path="data/wikisql/dev.jsonl"
test_questions_path="data/wikisql/test.jsonl"


retriever="dense"  #[lexical, dense]
max_row_num=5
max_cell_len=-1

process_table_corpus="--process_table_corpus" #--process_table_corpus
process_questions="--process_questions"  #--process_questions

processed_tables_path="data/wikisql/processed/processed_tables_for_"${retriever}"_row_"${max_row_num}"_max_cell_len_"$max_cell_len".json"

# train set
python preprocess/wikisql_preprocess.py $process_table_corpus $process_questions $with_value --tables_path $tables_path --questions_path $train_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_tables_path --max_row_num $max_row_num --max_cell_len $max_cell_len

# dev set
python preprocess/wikisql_preprocess.py $process_questions $with_value --tables_path $tables_path --questions_path $dev_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_tables_path --max_cell_len $max_cell_len

# test set
python preprocess/wikisql_preprocess.py $process_questions $with_value --tables_path $tables_path --questions_path $test_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_tables_path --max_cell_len $max_cell_len
