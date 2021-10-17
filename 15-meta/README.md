# Meta Learning

## Basic Concepts

* `MAML`'s target is to **find good initialization** for **a specific model with specific task**, eg. CNN. Therefore, it will take CNN as $F$ to update.

* When using default `forward()` function, parameters will be added to gradient graph. Therefore, if the original parameters shouldn't be changed, we should

  * duplicate the model and use new one to compute

  * realize another forward function
  
    > This method can **preserve** gradient graph for **outer update** on original model.
  
    ```python
    class Classifier(nn.Module):
        def __init__(self, in_ch, k_way):
            super(Classifier, self).__init__()
            self.conv1 = ConvBlock(in_ch, 64)
            self.conv2 = ConvBlock(64, 64)
            self.conv3 = ConvBlock(64, 64)
            self.conv4 = ConvBlock(64, 64)
            self.logits = nn.Linear(64, k_way)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.shape[0], -1)
            x = self.logits(x)
            return x
    
        def functional_forward(self, x, params):
            '''
            Arguments:
            x: input images [batch, 1, 28, 28]
            params: model parameters, 
                    i.e. weights and biases of convolution
                         and weights and biases of 
                                       batch normalization
                    type is an OrderedDict
    
            Arguments:
            x: input images [batch, 1, 28, 28]
            params: The model parameters, 
                    i.e. weights and biases of convolution 
                         and batch normalization layers
                    It's an `OrderedDict`
            '''
            for block in [1, 2, 3, 4]:
                x = ConvBlockFunction(
                    x,
                    params[f'conv{block}.0.weight'],
                    params[f'conv{block}.0.bias'],
                    params.get(f'conv{block}.1.weight'),
                    params.get(f'conv{block}.1.bias'))
            x = x.view(x.shape[0], -1)
            x = F.linear(x,
                         params['logits.weight'],
                         params['logits.bias'])
            return x
    ```
  
* training loops of `MAML`

  * outer loop: separate original dataset into **tasks**

    * updating original model is within Meta Algorithm

    ```python
    for step in tqdm(range(len(train_loader) // meta_batch_size)): 
        x, train_iter = get_meta_batch(
            meta_batch_size, k_shot, q_query, 
            train_loader, train_iter)
        meta_loss, acc = MetaAlgorithm(
            meta_model, optimizer, x, 
            n_way, k_shot, q_query, loss_fn)
        train_meta_loss.append(meta_loss.item())
        train_acc.append(acc)
    ```

    * inner loop: separate **a task** into support set and query set

      ```python
      for meta_batch in x:
          support_set = meta_batch[: n_way * k_shot]  
          query_set = meta_batch[n_way * k_shot :]    
      
          fast_weights = OrderedDict(model.named_parameters())
      
          ### ---------- INNER TRAIN LOOP ---------- ###
          for inner_step in range(inner_train_step): 
              train_label = create_label(n_way, k_shot).to(device)
              logits = model.functional_forward(support_set, fast_weights)
              loss = criterion(logits, train_label)
      
              fast_weights = inner_update(fast_weights, loss, inner_lr)
      ```

      * training and validation data are already separated when using `cat` in `get_meta_batch`, just like cut tail of something and concatenate it from another direction.

        ```python
        train_data = (task_data[:, :k_shot].reshape(-1, 1, 28, 28))
        val_data = (task_data[:, k_shot:].reshape(-1, 1, 28, 28))
        task_data = torch.cat((train_data, val_data), 0)
        ```

* When testing final `MAML` performance, i.e., the initialization quality, we firstly fine-tune the model for some inner loops and then validate it as test.

  ```python
  test_acc = []
  for test_step in tqdm(range(
          len(test_loader) // (test_batches))):
      x, test_iter = get_meta_batch(
          test_batches, k_shot, q_query, 
          test_loader, test_iter)
      # When testing, we update 3 inner-steps
      _, acc = MetaAlgorithm(meta_model, optimizer, x, 
                    n_way, k_shot, q_query, loss_fn, 
                    inner_train_step=3, train=False)
      test_acc.append(acc)
  print("  Testing accuracy: ", "%.3f %%" % (np.mean(test_acc) * 100))
  ```

  