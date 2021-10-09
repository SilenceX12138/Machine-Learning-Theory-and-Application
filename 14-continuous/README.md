# Life-long Learning

## Basic Concept

* dictionary generator

  ```python
  self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #extract all parameters in models
  ```

* underscore in Python: https://zhuanlan.zhihu.com/p/36173202

  * `_var`: only available within range, like private a class method.
  * `var_`: avoid conflict with keywords
  * `__var`: name mangling, avoid overwrite when defining subclass.
  * `__var__`: dunder methods

* `dim` in sum and mean

  * sum

    ```python
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    ```

    * `dim=0`: output of `[1, 4]` and reduced to `[4]`
    * `dim=1`: output of `[4, 1]` and reduced to `[4]`

  * mean

    ```python
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> torch.mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    ```

    * `dim=0`: output of `[1, 4]` and reduced to `[4]`
    * `dim=1`: output of `[4, 1]` and reduced to `[4]`