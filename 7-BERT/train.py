from transformers import BertForQuestionAnswering, BertTokenizerFast

from utils.data import get_dataloader
from utils.env import enable_reproduce, get_device
from utils.model import model_function


def train():
    enable_reproduce()
    # You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    train_loader = get_dataloader('train', tokenizer)
    dev_loader = get_dataloader('dev', tokenizer)
    model_function(train_loader, dev_loader, model, tokenizer)


if __name__ == '__main__':
    train()
