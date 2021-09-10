# Anime Face Generation

## Basic Concepts

* Usually, the discriminator needs training for **more** times than generator.

  * `n_critic`: train generator once after training discriminator `n_critic` times

* `view(-1)`: wrap up all dimensions into `1`.

  * eg. Given y.shape as `[64, 1, 1, 1]`, and then shape of `y.view(-1)` is `[64]`.

* special `Conv2d` parameters

  * half: $s=2\ and\ p=(k-1)/2$
  * equal: $k=s=1$​

* Upsample

  ```python
  nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
  ```

  * double: $s=2\ and\ p=(k-1)/2\ and\ output\_padding=1$​
  
    > **Different place from half `Conv2d` is `output_padding`.**
  
* When showing images, `torchvision` images needs permuting to place `channel` dimension at the **last** position.

  ```python
  plt.imshow(grid_img.permute(1, 2, 0))
  ```

* When saving images, `rgb` values needs mapping to $[0, 255]$.

  ```python
  torchvision.utils.save_image(imgs_sample[i], f'output/{i+1}.jpg')
  ```

## WGAN Realization

* remove last layer of `Sigmoid` in original Discriminator

  ```python
  nn.Conv2d(in_dim, dim, 5, 2, 2),
  nn.LeakyReLU(0.2),
  conv_bn_lrelu(dim, dim * 2),
  conv_bn_lrelu(dim * 2, dim * 4),
  conv_bn_lrelu(dim * 4, dim * 8),
  nn.Conv2d(dim * 8, 1, 4),
  # nn.Sigmoid(),
  ```

* change loss functions of generator and discriminator

  ```python
  loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
  loss_G = -torch.mean(D(f_imgs))
  ```

* clip discriminator's parameters

  ```python
  opt_D.step()
  for p in D.parameters():
     p.data.clamp_(-config.clip_value, config.clip_value)
  ```

* use non-momentum optimizer like `RMSProp`

  ```python
  opt_D = torch.optim.RMSprop(D.parameters(), lr=config.lr)
  opt_G = torch.optim.RMSprop(G.parameters(), lr=config.lr)
  ```

## Result Display

* **Epoch1**

  <img src="https://i.loli.net/2021/09/10/EyiYNZjr7Hzcx2Q.jpg" alt="Epoch_001" style="zoom: 50%;" />

* **Epoch10**

  <img src="https://i.loli.net/2021/09/10/RH7WuTKSzNUwgV5.jpg" alt="Epoch_010" style="zoom: 50%;" />

* **Epoch20**

  <img src="https://i.loli.net/2021/09/10/3gV8RJO5j4WGTkF.jpg" alt="Epoch_020" style="zoom: 50%;" />

* **Epoch30**

  <img src="https://i.loli.net/2021/09/10/BZqXE6ohV5zvSJY.jpg" alt="Epoch_030" style="zoom: 50%;" />

* **Epoch40**

  <img src="https://i.loli.net/2021/09/10/aneqLGuNWR7PpM3.jpg" alt="Epoch_040" style="zoom: 50%;" />

* **Epoch50**

  <img src="https://i.loli.net/2021/09/10/KkQhGXq9R4EnwIf.jpg" alt="Epoch_050" style="zoom: 50%;" />