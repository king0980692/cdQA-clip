from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.evaluation import evaluate_pipeline, evaluate_reader
from cdqa.utils.download import *
from cdqa.pipeline import QAPipeline

import json
from ast import literal_eval
import pandas as pd
import torch


def test_evaluate_pipeline():

    # download_bnpp_data("./data/bnpp_newsroom_v1.1/")
    # download_model("bert-squad_1.1", dir="./models")
    df = pd.read_csv(
        "./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv",
        converters={"paragraphs": literal_eval},
    )
    df = filter_paragraphs(df)



    cdqa_pipeline = QAPipeline(reader="./models/bert_qa.joblib", 
                               n_jobs=4)

    cdqa_pipeline.fit_retriever(df)

    eval_dict = evaluate_pipeline(cdqa_pipeline, 
                                  "data/SQuAD_1.1/dev-v1.1.json", 
                                  output_dir=None)
    
    print(eval_dict["exact_match"])
    print(eval_dict["f1"])

    # assert eval_dict["exact_match"] > 0.8
    # assert eval_dict["f1"] > 0.8


def test_evaluate_reader():

    # download_model("bert-squad_1.1", dir="./models")
    cdqa_pipeline = QAPipeline(reader="./models/bert_qa.joblib", 
                               n_jobs=-1)

    eval_dict = evaluate_reader(cdqa_pipeline, 
                                "data/SQuAD_1.1/dev-v1.1.json"
                                )

    print(eval_dict["exact_match"])
    print(eval_dict["f1"])
    # assert eval_dict["exact_match"] > 0.8
    # assert eval_dict["f1"] > 0.8

if __name__ == '__main__':
    test_evaluate_pipeline()
    # test_evaluate_reader()
