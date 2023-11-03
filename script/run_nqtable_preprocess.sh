dataset="nqtable"
partial_tables_path="data/nqtable/tables/partial_tables.jsonl"
tables_path="data/nqtable/tables/tables.jsonl"
train_questions_path="data/nqtable/interactions/train.jsonl"
dev_questions_path="data/nqtable/interactions/dev.jsonl"
test_questions_path="data/nqtable/interactions/test.jsonl"

# collect the tables used in train, dev and test set for quick training and evaluation
python preprocess/collect_nqtable_partial_tables.py --train_path $train_questions_path --dev_path $dev_questions_path --test_path $test_questions_path --partial_tables_path $partial_tables_path

retriever="dense" #[dense]
max_row_num=-1
max_cell_len=-1
output_dir="data/nqtable/processed/"
processed_tables_path="data/nqtable/processed/processed_tables_for_"${retriever}"_row_"${max_row_num}"_max_cell_len_"$max_cell_len".json"
processed_partial_tables_path="data/nqtable/processed/processed_partial_tables_for_"${retriever}"_row_"${max_row_num}"_max_cell_len_"$max_cell_len".json"

# process partial_tables
python preprocess/nqtable_preprocess.py --process_table_corpus $with_value $append_title --tables_path $partial_tables_path --questions_path $train_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_partial_tables_path   --max_row_num $max_row_num --dataset $dataset --max_cell_len $max_cell_len

# process global tables
python preprocess/nqtable_preprocess.py --process_table_corpus $with_value $append_title --tables_path $tables_path --questions_path $train_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_tables_path   --max_row_num $max_row_num --dataset $dataset --max_cell_len $max_cell_len

# train set
python preprocess/nqtable_preprocess.py --process_questions $with_value --tables_path $tables_path --questions_path $train_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_tables_path   --max_row_num $max_row_num --dataset $dataset --max_cell_len $max_cell_len

# dev set
python preprocess/nqtable_preprocess.py --process_questions $with_value --tables_path $tables_path --questions_path $dev_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_tables_path   --dataset $dataset --max_cell_len $max_cell_len

# test set
python preprocess/nqtable_preprocess.py --process_questions $with_value --tables_path $tables_path --questions_path $test_questions_path --output_dir $output_dir --retriever $retriever --processed_tables_path $processed_tables_path   --dataset $dataset --max_cell_len $max_cell_len
