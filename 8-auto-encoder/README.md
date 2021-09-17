# Anomaly Detection

## Basic Concept

* calculation of `ROC AUC`

  <img src="https://i.loli.net/2021/09/17/DZSiYAC76G5PhqM.png" alt="image-20210917140544726" style="zoom: 45%;" />

  * **sort** all data by prediction value
  * set threshold along with the **gap** of values
  * calculate `tp` and `fp` under all situations
  * plot `ROC` curve
  * calculate the area

* build working directory

  * use `os.path.join` to get path instead of explicitly adding `/` in the string

    ```python
    workspace='.'
    data_path='data/dataset' # '/' can be added within the path
    
    os.makedirs(os.path.join(workspace, data_path), exist_ok=True)
    ```

  * `os.makedirs()` can make directories **reversely**.

    ```python
    os.makedirs('./test1/test2/test3/test4')
    ```

* Windows doesn't support lambda transform in `torchvision` when enabling **multiprocessing**.

  ```python
  self.transform = transforms.Compose([
      transforms.Lambda(lambda x: x.to(torch.float32)),
      transforms.Lambda(lambda x: 2. * x / 255. - 1.),
      # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
  ])
  ```

  * Solution: set `num_workers=0` in dataloader.

    ```python
    test_loader = DataLoader(test_dataset,
                             sampler=test_sampler,
                             batch_size=test_config.batch_size,
                             num_workers=0)
    ```

* VAE: https://blog.csdn.net/weixin_43876801/article/details/103654186

* GAN&Auto-Encoder

  <img src="https://pic2.zhimg.com/80/v2-c02e17ca61a91ab7fbfa1d52be1a05f8_1440w.jpg?source=1940ef5c" alt="img" style="zoom: 33%;" />

  <img src="https://pic1.zhimg.com/80/v2-6955c77ccb68b4ac464848dc9be0fd5f_1440w.jpg?source=1940ef5c" alt="img" style="zoom:33%;" />