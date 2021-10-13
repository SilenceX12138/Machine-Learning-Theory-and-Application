# Network Compression

## Basic Concepts

* In all forms of convolution, **1 filter** generates **1 channel** of feature map, i.e., a feature.

  * a feature is 2D in image and 3D in video

  <img src="https://i.imgur.com/Hqhg0Q9.png" alt="img" style="zoom: 67%;" />

* group convolution(GC) requires `group_size` divisible by `in_channels` and `out_channels`, because only in this way can we decide the channels of kernel in each group.

  * modify a `(12*10*10)` matrix into `(36*8*8)` with GC

    ```python
    a = torch.rand(1,12,10,10)
    m = nn.Conv2d(12, 36, 3, stride=2, padding=1, groups=4) # 4 2D-kernels of size 9*3*3
    m(a)
    ```

* number of parameters in convolution

  * $DW=in\_channels*kernel\_size^2$
  * $PW=in\_channels*out\_channels$

## Implementation

* Knowledge distillation is **not** an independent method. For instance, it can be combined with other algorithms like network architecture design.

* loss of student network: $Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{Original Loss})$

  ```python
  def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5):
      hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 
      soft_loss = alpha * 9 * F.kl_div(outputs.softmax(dim=-1) / 3, teacher_outputs.softmax(dim=-1) / 3)
      return hard_loss + soft_loss
  ```


