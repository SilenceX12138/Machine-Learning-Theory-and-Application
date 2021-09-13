# Question Answering with BERT

## Basic Concept

* automatic mixed precision training (fp16): https://zhuanlan.zhihu.com/p/103685761

* `ids` is short for `indices`, which means indices of input sequence tokens in the vocabulary.

  * `[CLS]`: 101
  * `[SEP]`: 102

* `torch.argmax(input, dim)`: `dim` stands for the dimension to be **reduced**

  ```python
  >>> a
  tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
          [-0.7401, -0.8805, -0.3402, -1.1936],
          [ 0.4907, -1.3948, -1.0691, -0.3132],
          [-1.6092,  0.5419, -0.2993,  0.3195]])
  >>> torch.argmax(a, dim=1)
  tensor([ 0,  2,  0,  1])
  ```

  * This is the same as the **second** return value of `torch.max(input, dim)`.

* split windows and `doc_stride`

  * train set: In order to decrease size of input paragraph, we can select only part of the whole paragraph, which contains the answer.
  * valid/test set: Due to the fact that we have no idea where the answer is, the only solution is set a length of every possible part of paragraph(`doc_stride`)
  * After selecting parts, `padding` is needed to make sure all input data has the same length as `max_paragraph_len`.

  > Because tokenizing a batch cannot split data set as above, so we can only implement our own data set.
  >
  > ```python
  > pt_batch = tokenizer(
  >     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
  >     padding=True,
  >     truncation=True,
  >     max_length=512,
  >     return_tensors="pt"
  > )
  > ```

