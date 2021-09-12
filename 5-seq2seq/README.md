# Machine Translation

## Basic Concept

* BLEU: https://blog.csdn.net/qq_31584157/article/details/77709454
* fairseq: https://zhuanlan.zhihu.com/p/361835267
* beam seach: https://zhuanlan.zhihu.com/p/28048246
* teacher forcing: to train the model to predict the next token based on prefix, we feed the right shifted target sequence as the decoder input.

## Implement

### Reproductivity

* PyTorch random number generator

  ```python
  import torch
  torch.manual_seed(0)
  ```

* Python

  ```python
  import random
  random.seed(0)
  ```

* Random number generators in other libraries

  ```python
  import numpy as np
  np.random.seed(0)
  ```

* CUDA convolution benchmarking

  ```python
  torch.backends.cudnn.benchmark = False
  ```

  > https://pytorch.org/docs/stable/notes/randomness.html

* Avoiding nondeterministic algorithms

  ```python
  # torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  ```

  > `torch.use_deterministic_algorithms()` which will make other PyTorch operations behave deterministically.
  
* Dataloader

  ```python
  def seed_worker(worker_id):
      worker_seed = torch.initial_seed() % 2**32
      numpy.random.seed(worker_seed)
      random.seed(worker_seed)
  
  g = torch.Generator()
  g.manual_seed(0)
  
  DataLoader(
      train_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      worker_init_fn=seed_worker
      generator=g,
  )
  ```

  > DataLoader will reseed workers following [Randomness in multi-process data loading](https://pytorch.org/docs/stable/data.html#data-loading-randomness) algorithm. Use `worker_init_fn()` and generator to preserve reproducibility:

### Dataloading

#### TranslationTask

- used to load the **binarized** data created above
- well-implemented data iterator (`dataloader`)
- built-in `task.source_dictionary` and `task.target_dictionary` are also handy
- well-implemented beam search decoder

```python
from fairseq.tasks.translation import TranslationConfig, TranslationTask

## setup task
task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)
```

#### sentencepiece

```python
task.target_dictionary.string(
    sample['target'],
    config.post_process,
)
```

#### Dataset Iterator

```python
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond 
        # first call of this method has no effect. 
    )
    return batch_iterator

demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
sample
```

- Controls every batch to contain no more than N tokens, which optimizes GPU memory efficiency

- Shuffles the training set for every epoch

- Ignore sentences exceeding maximum length

- Pad all sentences in a batch to the same length, which enables parallel computing by GPU

- Add `eos` and shift one token

  - teacher forcing

  - Generally, prepending `bos` to the target would do the job (as shown below).<img src="https://i.imgur.com/0zeDyuI.png" alt="seq2seq" style="zoom:67%;" />

  - in `fairseq` however, this is done by moving the `eos`token to the beginning. Empirically, this has the same effect. For instance:

    ```
    # output target (target) and Decoder input (prev_output_tokens): 
                 eos = 2
              target = 419,  711,  238,  888,  792,   60,  968,    8,    2
    prev_output_tokens = 2,  419,  711,  238,  888,  792,   60,  968,    8
    ```

  - each batch is a **python dict**, with string key and Tensor value. Contents are described below:

    ```
    batch = {
      "id": id, # id for each example 
      "nsentences": len(samples), # batch size (sentences)
      "ntokens": ntokens, # batch size (tokens)
      "net_input": {
          "src_tokens": src_tokens, # sequence in source language
          "src_lengths": src_lengths, # sequence length of each example before padding
          "prev_output_tokens": prev_output_tokens, # right shifted target, as mentioned above.
      },
      "target": target, # target sequence
    }
    ```

#### Model

* How to use transformer: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

  ```python
  transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
  src = torch.rand((10, 32, 512))
  tgt = torch.rand((20, 32, 512))
  out = transformer_model(src, tgt)
  ```

  
