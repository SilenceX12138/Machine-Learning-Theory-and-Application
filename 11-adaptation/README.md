# Domain Adaptation

## Basic Concept

* methods to clear gradient

  * `optimizer.zero_grad()`

  * `model.zero_grad()`

    ```python
    G.zero_grad()
    ```

* method to backward: `loss.backward()`

  ```python
  loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
  D.zero_grad()
  loss_D.backward()
  
  loss_G = -torch.mean(D(f_imgs))
  G.zero_grad()
  loss_G.backward()
  ```

  > `step` can only be executed by optimizer, and `backward` can only be executed by criterion.

* `Variable` still needs explicitly set `requires_grad` to `True`

  ```python
  a = Variable(tensor, requires_grad=True)
  ```

  * use `torch.tensor` can also define a tensor with gradient

    ```python
    torch.tensor([1.,2.,3.], requires_grad=True)
    ```

    * Note that `int` data type **cannot** hold gradients, so the element should be float or complex data type.

* When calling `backward()`, only **scalar** element is legal.

  ```python
  x = torch.tensor([1., 2., 3.], requires_grad=True)
  y = torch.tensor([4., 5., 6.], requires_grad=True)
  z = x**2 + y**2
  f = x + y + z
  s1 = 6 * f.sum()
  s2 = f.sum()
  ```

  * `sum()` doesn't influence calculation of gradients due to the fact that **only values instead of loss output** will be used to get gradients.

  * loss composite is not easily sum up. Instead, the whole calculation path should be copied.

    ```python
    loss1 = domain_criterion(domain_logits, domain_label)
    loss2 = class_criterion(class_logits, source_label) \
           - train_config.lamb * domain_criterion(domain_logits, domain_label)
    ```

    * `loss2` **cannot** be set with `loss1`

  * use `detech()` when some variables **don't** need gradients

    ```python
    feature = extractor(mixed_data)
    # We don't need to train feature extractor in step 1.
    # Thus we detach the feature neuron to avoid backpropgation.
    domain_logits = classifier(feature.detach())
    ```

* When `backward() ` has two different paths, set `retain_graph=True` for multi-times backward.

  ```python
  s1.backward(retain_graph=True)
  ```

  * Gradients from different paths will be added up.

* implement DIY `autograd` function

  ```python
  class GRL(Function):
      @staticmethod
      def forward(self, i):
          return i
  
      @staticmethod
      def backward(self, grad_output):
          grad_output = grad_output.neg()
          return grad_output
      
  s = GRL.apply(s)
  ```

  * `GRL` is added **behind** s, and it can be taken as an activation function.
    * https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    
  * use `GRL` can train `DaNN` quickly by optimizing Discriminator and Extractor simutaneously
  
  * module with `GRL` should be trained later to avoid influencing backward of other modules
  
    ```python
    # Step 1 : train feature extractor and label classifier
    feature = extractor(mixed_data)
    class_logits = predictor(feature[:source_data.shape[0]])
    domain_logits = classifier(feature)
    loss = class_criterion(class_logits, source_label)
    running_F_loss += loss.item()
    loss.backward(retain_graph=True)
    
    # Step 2 : train domain classifier
    # classifier has to be trained later due to GRL
    feature = GRL.apply(feature)
    domain_logits = classifier(feature)
    loss = domain_criterion(domain_logits, domain_label)
    running_D_loss += loss.item()
    loss.backward()
    ```
  
  * `GRL` training process is **slower** than `GAN` method, and the performance is not as good as `GAN`.

* Domain Knowledge: Pre-process source data can help extractor perform better. For instance, change RGB source into gray and extract edges can help when target is gray-scale edge photos.
* `BCE` is used when calculating binary classification loss.