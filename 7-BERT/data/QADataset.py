import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, split, questions, tokenized_questions,
                 tokenized_paragraphs):
        self.split = split  # split means mode: train/dev/test
        self.questions = questions  # list of questions(dict)
        self.tokenized_questions = tokenized_questions  # list of tokenized question text
        self.tokenized_paragraphs = tokenized_paragraphs  # list of tokenized paragraph
        self.max_question_len = 40
        self.max_paragraph_len = 150

        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 150

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP] = 193
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[
            question["paragraph_id"]]
        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph
            answer_start_token_pos = tokenized_paragraph.char_to_token(
                question["answer_start"])
            answer_end_token_pos = tokenized_paragraph.char_to_token(
                question["answer_end"])
            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token_pos + answer_end_token_pos) // 2
            paragraph_start = max(
                0,
                min(mid - self.max_paragraph_len // 2,
                    len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] \
                                       + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start:paragraph_end] \
                                       + [102]
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window
            answer_start_token_pos += len(input_ids_question) - paragraph_start
            answer_end_token_pos += len(input_ids_question) - paragraph_start
            # Pad sequence and obtain inputs to model
            input_ids, token_type_ids, attention_mask = self.padding(
                input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
                   answer_start_token_pos, answer_end_token_pos

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                # Slice paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] \
                                           + [102] # question cannot be sliced
                input_ids_paragraph = tokenized_paragraph.ids[i:i + self.max_paragraph_len] \
                                           + [102]
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(
                    input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), \
                   torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) \
                                       - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph \
                                       + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) \
                                                       + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) \
                                                        + [0] * padding_len
        return input_ids, token_type_ids, attention_mask
