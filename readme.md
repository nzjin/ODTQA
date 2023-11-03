The dataset and processing code for IJCNLP-AACL 2023 paper “[Enhancing Open-Domain Table Question Answering via Syntax- and Structure-aware Dense Retrieval](https://arxiv.org/abs/2309.10506)”.

### Prepare dataset

**1. Download the original dataset**

Please download the original [WikiSQL](https://github.com/salesforce/WikiSQL) and [NQ-Tables](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md) . Then decompress and move them to data/wikisql and data/nqtables folders, respectively.

```
data
└───nqtable
│   └───interactions
│       │   ...
|   └───tables
|       |   ...
│   
└───wikisql
    │   dev.db
    │   dev.jsonl
    |   dev.tables.jsonl
    |   ...
```

**2. preprocess the original dataset**

```bash
cd ODTQA
bash script/run_wikisql_preprocess.sh
bash script/run_nqtable_preprocess.sh
```

The processed datasets for retrieval will be saved in nqtable/processed folder and wikisql/processed folder.

Mention: you can also download the processed dataset directly from [here](https://drive.google.com/file/d/1PqOcQJakTVRxnTLYzO7kahNY9d34dXwR/view?usp=drive_link).
