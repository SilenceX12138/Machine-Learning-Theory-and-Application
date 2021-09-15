import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertForQuestionAnswering, BertTokenizerFast

from utils.data import get_dataloader
from utils.env import get_device
from utils.model import translate


def eval():
    print("Evaluating Test Set ...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    test_loader = get_dataloader('test', tokenizer)
    device, _ = get_device()
    model = BertForQuestionAnswering.from_pretrained('./checkpoints').to(device)
    pred(test_loader, model, tokenizer, device)


def pred(test_loader: DataLoader, model: nn.Module, tokenizer, device):
    model.eval()
    result = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            # output: 4(window_count)*193
            output = model(input_ids=data[0].squeeze(dim=0).to(device), # With squeeze, model will take 4 as batch size.
                           token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(translate(data, output, tokenizer))
    save_pred(test_loader, result)


def save_pred(test_loader: DataLoader, result: list):
    result_file = "prediction.csv"
    with open(result_file, 'w') as f:
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_loader.dataset.questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")


if __name__ == '__main__':
    eval()
