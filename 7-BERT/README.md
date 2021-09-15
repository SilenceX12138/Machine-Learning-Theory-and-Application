# Question Answering with BERT

## Basic Concept

* automatic mixed precision training (fp16): https://zhuanlan.zhihu.com/p/103685761

* `ids` is short for `indices`, which means indices of input sequence tokens in the BERT training vocabulary.

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

* tokenize data

  * Because tokenizing a batch cannot split data set as above, so we can only implement our own data set.

    ```python
    pt_batch = tokenizer(
     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
     padding=True,
     truncation=True,
     max_length=512,
     return_tensors="pt",
     add_special_tokens=False
    )
    ```

    * This **will not** combine sentences in list like question and paragraph.

  * Also, `tokenizer` can combine question and paragraph automatically.

    ```python
    question = '李宏毅幾班?'
    paragraph = '李宏毅幾班大金。'
    encoded = chi_tokenizer.encode(question, paragraph) # encoded is just a list instead of input data
    decoded = chi_tokenizer.decode(encoded)
    print(encoded)
    print(decoded)
    ```

  * Use `tokenizer` to generate input: `transformers.tokenization_utils_base.BatchEncoding`

    ```python
    inputs = chi_tokenizer(question, paragraph, return_tensors='pt')
    ```

* split windows and `doc_stride`

  * train set: In order to decrease size of input paragraph, we can select only part of the whole paragraph, which contains the answer.
  * valid/test set: Due to the fact that we have no idea where the answer is, the only solution is set a length of every possible part of paragraph(`doc_stride`)
  * After selecting parts, `padding` is needed to make sure all input data has the same length as `max_paragraph_len`.
  
* `batch_size` can only be 1 when validating and testing

  > As long as paragraph length is larger than expected, windows needs to be split. Therefore, different paragraphs may have varied count of windows, which cannot be combined into one batch. Also, the model will take count of windows as batch size after squeezing the first dimension of input.

* `AdamW` optimizer: https://www.jianshu.com/p/e17622b7ffee

  > This is the optimizer training BERT.

* When iterate the data loader, each tensor in the `__getitem()_` return tuple will be added a dimension of `batch_size`

  * `data = [[1, 2, 3]]` as list will be `data[0] as batch_size*3`. Note that list **won't** be given a dimension of `batch_size`.

  * When input it into the model, we don't need to consider the `batch_size` dimension. However, when handling them manually, this dimension cannot be neglected.

    * First dimension of input data **is always batch size** in PyTorch.

      * original batch size

        ```python
        output = model(input_ids=data[0],
                       token_type_ids=data[1],
                       attention_mask=data[2],
                       start_positions=data[3],
                       end_positions=data[4])
        ```

      * batch size is **window count**

        ```python
        output = model(input_ids=data[0].squeeze(dim=0).to(device), 
                       token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        ```


## Optimization

* learning rate decay: https://www.jianshu.com/p/9643cba47655

  ```python
  model = [Parameter(torch.randn(2, 2, requires_grad=True))]
  optimizer = SGD(model, 0.1)
  scheduler = ExponentialLR(optimizer, gamma=0.9)
  
  for epoch in range(20):
      for input, target in dataset:
          optimizer.zero_grad()
          output = model(input)
          loss = loss_fn(output, target)
          loss.backward()
          optimizer.step()
      scheduler.step() # update learning rate every epoch instead of batch
  ```

  > Even with adaptive learning rate, `Adam` still needs learning rate decay when training.

* slice list: Even when index is beyond range, only the edge value will be included instead of popping up an error.

  ```python
  l = [1, 2, 3]
  l[1:1000] # [2, 3]
  ```

* The output starting and ending positions of answer may not be `starting<ending`, so it needs manually fixing.

  * find ending index after starting position
