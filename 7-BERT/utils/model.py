import torch
import torch.nn as nn
from config import train_config
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW

from utils.env import get_device


def model_function(train_loader: DataLoader, dev_loader: DataLoader,
                   model: nn.Module, tokenizer):
    device, accelerator = get_device()
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate)
    if train_config.fp16_training:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader)

    print("Start Training ...")
    model.to(device)
    model.train()
    for epoch in range(train_config.num_epoch):
        step = 1
        train_loss = train_acc = 0
        for data in tqdm(train_loader):
            # Load all data into GPU
            data = [i.to(device) for i in data]
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
            output = model(input_ids=data[0],
                           token_type_ids=data[1],
                           attention_mask=data[2],
                           start_positions=data[3],
                           end_positions=data[4])
            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) &
                          (end_index == data[4])).float().mean()
            train_loss += output.loss
            if train_config.fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            ##### TODO: Apply linear learning rate decay #####
            # Print training loss and accuracy over past logging step
            if step % train_config.logging_step == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / train_config.logging_step:.3f}, acc = {train_acc / train_config.logging_step:.3f}"
                )
                train_loss = train_acc = 0
    if train_config.validation:
        validate(model, tokenizer, epoch, dev_loader, device)


def validate(model: nn.Module, tokenizer, epoch: int, dev_loader: DataLoader,
             device):
    print("Evaluating Dev Set ...")
    model.eval()
    with torch.no_grad():
        dev_acc = 0
        for i, data in enumerate(tqdm(dev_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(device),
                           token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
            # prediction is correct only if answer text exactly matches
            dev_acc += translate(data, output, tokenizer) \
                       == dev_loader.dataset.questions[i]["answer_text"]
        print(
            f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}"
        )
    model.train()


def translate(data, output, tokenizer):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index:end_index + 1])
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ', '')
