# Adversarial Attack

## Basic Concept

* `torchvision.transforms.ToTensor`: Converts a `PIL`Image or `numpy.ndarray` $(H * W * C)$​​ in the range $[0, 255]$​​​​ to a `torch.FloatTensor` of shape $(C * H * W)$​ in the range $[0.0, 1.0]$​​​​.

* When implementing `i-fgsm`, the step size **cannot** be set equal to epsilon, or the `x_adv` will always be on the corner of square $2*\epsilon$.

* `torch` functions related to `dim`
  * `squeeze` and `unsqueeze`: https://blog.csdn.net/wwwlyj123321/article/details/88972717
  * `argmax`: https://blog.csdn.net/qq_27261889/article/details/88613932
  
* `os.path.relpath`: return a relative file path to path either from the `current` directory or from an optional `start` directory

* clip value into $[min,\ max]$
  * ```python
    torch.min(torch.max(x, min_value), max_value)
    ```
  
  * ```python
    x.clamp(min, max)
    torch.clamp(a, min=-0.5, max=0.5)
    ```
  
    > `min` and `max` are optional values in `clamp()`

* methods to change tensor shape: https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/

* `numpy.r_`: Stack arrays along specified axis, default to the **first** one.

  ```python
  a = np.array([[0, 1, 2], [3, 4, 5]])
  np.r_['-1', a, a] # concatenate along last axis
  # array([[0, 1, 2, 0, 1, 2],
  #        [3, 4, 5, 3, 4, 5]])
  np.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, dim>=2
  # array([[1, 2, 3],
  #        [4, 5, 6]])
  ```

* When `softmax` along `dim`, it means to calculate within `cnt` values of `dim`.

  ```python
  >>> m = nn.Softmax(dim=1)
  >>> input = torch.randn(2, 3)
  >>> input
  tensor([[ 1.3312, -0.4179, -0.0077],
          [-0.0984, -0.9152, -1.4912]])
  >>> output = m(input)
  >>> output
  tensor([[0.6963, 0.1211, 0.1825],
          [0.5916, 0.2614, 0.1470]])
  ```
  
* When using `*` to conduct **element-wise** multiplying between `a` and `b`, below qualifications need satisfying.

  * Take `a` as **front** matrix

  * `b` shape has to be the subset of `a` trailing dimensions. For instance, `a` has $(8*3*32*32)$, and then `b` has to be:

    * `1D`: $(32)/(1)$
    * `2D`: $(32*32)/(32*1)/(1*32)/(1*1)$
    * `3D`: $(3*32*32)/(3*32*1)/(3*1*32)/(3*1*1)/(1*1*1)$

    > broadcast calculation: https://pytorch.org/docs/stable/notes/broadcasting.html



