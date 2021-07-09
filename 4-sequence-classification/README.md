# Speaker Classification

## Sequence Data

* When training, the dataset needs to **adjust all data** within the same batch to a **settled** segment length.

  > However, in order to use more features, the testing procedure **doesn't need** length adjustment.

  * cut long data

    ```python
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))
    
        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
    ```

  * pad short data

    ```python
    def collate_batch(batch):
        # Process features within a batch.
        """Collate a batch of data."""
        mel, speaker = zip(*batch)
        # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
        mel = pad_sequence(
            mel, batch_first=True,
            padding_value=-20)  # pad log 10^(-20) which is very small value.
        # mel: (batch size, length, 40)
        return mel, torch.FloatTensor(speaker).long()
    ```

    * `collate_fn` is adjusted because we need to pad sequence data.
    * `zip(*batch)` change `(data, target)` tuples into list of data and list of target.

  * define `dataloader`

    ```python
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    ```


## Transformer

* Q, K and V may stand for different meanings as following.

  * Matrix result of $W^q*X$

  * Word vectors, i.e., the input data.

    > That's why so many articles say that self-attention has $Q=K=V$. In traditional attention, the attention score is calculated with `source` and `target`.
    >
    > * difference between self-attention and attention: https://blog.csdn.net/qq_40585800/article/details/112427990

* `batch_first`: In most **sequence input** problems, the **batch** dimension should be adjusted at **second** place, such as `RNN`, `LSTM`, `Transformer`.

  ```python
  # out: (length, batch size, d_model)
  out = out.permute(1, 0, 2)  # reset sequence of dimensions
  # The encoder layer expect features in the shape of (length, batch size, d_model).
  # In version 1.9.0, parameter `batch_first` can be set.
  out = self.encoder_layer(out)
  # out = self.encoder(out)
  # out: (batch size, length, d_model)
  out = out.transpose(0, 1)  # exchange dim 0 and dim 1
  ```

* `d_model`&`channel`&`length`

  * `d_model` is the number of features in sequence data, and also the `channel`.
  * `length` is the number of timestamps within a data.

## Conformer

* When using `nn.Sequential`, the activation function should be a **module** instead of a **function**.

  ```python
  self.linear2 = nn.Sequential(
      nn.Linear(d_model, dim_feedforward),
      ReLU(),
      Dropout(dropout),
      nn.Linear(dim_feedforward, d_model),
      ReLU(),
      Dropout(dropout),
  )
  ```

  > In above case, `relu()` function will require an argument to be passed immediately.