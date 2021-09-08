# Semi-supervised Leaning with Pseudo Labels

## Basic Operations

* build a tensor with tensor list

  ```python
  torch.stack(tensor_list, dim=0)
  ```

  * `torch.cat` won't create a new dimension. Instead, it will concatenate all tensor in specific existing dimension.

    ```python
    l = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.cat((l, l), dim=0)
    '''
    tensor([[1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [4, 5, 6]])
    '''
    ```

* build a tensor with dimension `0`(a pure digit, like `torch.Size([])`)

  ```python
  torch.tensor(1)
  ```

  * `torch.Tensor` will create a list or a tensor with dimension `1`(scalar).

    ```python
    torch.Tensor(10) # a scalar with 10 random numbers as well as LongTensor()
    torch.Tensor([10]) # a scalar with 10. and dimension is torch.Size([1])
    ```

    > `Tensor` will take **all** int parameters as **size** parameters

  * `torch.tensor` infers the `dtype` automatically, while `torch.Tensor` returns a `torch.FloatTensor`.

    ```python
    >>> torch.Tensor([1, 2, 3]).dtype
    torch.float32
    >>> torch.tensor([1, 2, 3]).dtype
    Out[32]: torch.int64
    >>> torch.Tensor([True, False]).dtype
    torch.float32
    >>> torch.tensor([True, False]).dtype # True for 1 and False for 0
    torch.uint8
    ```

* create a `TensorDataset`

  ```python
  TensorDataset(data_tensor, label_tensor)
  ```

  > In all datasets, the `label_tensor` needs to be `Long` type.

## Update Training Set

* load unlabeled dataset

* predict with model

* select predictions beyond threshold to construct a new dataset

  * The best way is to establish a self dataset class, for `TensorDataset` will not lead the `transform` when loading data, which actually matters a lot to the performance of model.

    ```python
    def __getitem__(self, index):
            # transfrom the image every time when loading
            img = Image.open(self.data[index])
            if self.mode == 'train':
                return train_tfm(img), self.label[index]
            elif self.mode == 'semi':
                return train_tfm(img), self.label[index], index
            else:
                return test_tfm(img), self.label[index]
    ```

    > The experience tells us that `transform` should be done every time the data is accessed by the `DataLoader`.

  * Implement a method in self dataset class to sample data with **index**.

    ```python
    def sample_dataset(self, mode, index_list, label_list=None):
            """
            return a new FOOD11Dataset according to index
            """
            sample_data = [self.data[index] for index in index_list]
            if label_list is None:
                sample_label = [self.label[index] for index in index_list]
            else:
                sample_label = label_list
            return FOOD11DataSet(mode=mode, data=sample_data, label=sample_label)
    ```

* concatenate labeled dataset and pseudo label dataset

  * **DO NOT** use `torchvision` datasets with `TensorDataset` or self dataset.

    > The ways `torchvision` dataset is split into batches and the labels are loaded are different. Therefore,  `DataLoader` cannot handle this kind of mixed dataset.

## Adopt sealed models

* **REMEMBER TO CHANGE OUTPUT SIZE OF SEALED MODELS**

  ```python
  def set_parameter_requires_grad(model, feature_extract: bool):
      for param in model.parameters():
          param.requires_grad = feature_extract
  
  model = torchvision.models.resnet18(pretrained=True)
  set_parameter_requires_grad(model, False)
  model.fc = nn.Linear(512, num_classes)
  ```

  * When set `feature_extract` to `False`, only new output layer's parameters will be updated.

* `load_state_dict` **won't** change whether the parameter requires gradient.

* adjust `torchvision` modules: https://pytorch.apachecn.org/docs/1.0/finetuning_torchvision_models_tutorial.html