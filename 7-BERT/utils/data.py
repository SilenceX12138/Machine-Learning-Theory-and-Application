import json

from config import train_config
from torch.utils.data import DataLoader

from data.QADataset import QADataset


def read_data(file: str):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]


def get_dataloader(mode: str, tokenizer):
    question_list, paragraph_list = read_data(
        './data/dataset/hw7_{}.json'.format(mode))
    # When constructing data sets, tokenize questions and labels separately and combine them in `__getitme()__`
    token_question_list = tokenizer(
        [question["question_text"] for question in question_list],
        add_special_tokens=False)
    token_paragraph_list = tokenizer(paragraph_list, add_special_tokens=False)
    data_set = QADataset(mode, question_list, token_question_list,
                         token_paragraph_list)
    dataloader = DataLoader(
        data_set,
        batch_size=train_config.batch_size if mode == 'train' else 1,
        shuffle=(mode == 'train'),
        pin_memory=True)
    return dataloader
