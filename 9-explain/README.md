# Explainable ML

## Basic Concept

* judge whether a pre-trained model has already been fine-tuned

  * Head layers have similar embeddings of tokens
  * When in fine-tuned models, trailing layer will cluster tokens according to type, especially `question` embeddings. Also, same token of different types are usually at a far distance.
  
  | Bert Layer | Pre-trained                                                  | Fine-tuned                                                   |
  | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 1          | <img src="https://i.loli.net/2021/09/24/Dy5KTW1z2dl8LtN.png" alt="1" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/TWvKFP2murABfwX.png" alt="1" style="zoom:50%;" /> |
  | 2          | <img src="https://i.loli.net/2021/09/24/ZOasLzEReiw5Wvc.png" alt="2" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/KIM1LJFgUz5pQuv.png" alt="2" style="zoom:50%;" /> |
  | 3          | <img src="https://i.loli.net/2021/09/24/xFGZqXDiTcJ1YlV.png" alt="3" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/kTjHmN3p871MtvP.png" alt="3" style="zoom:50%;" /> |
  | 4          | <img src="https://i.loli.net/2021/09/24/ay5CMZAcS8dLzvJ.png" alt="4" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/FJ32fR7q9xLrsjZ.png" alt="4" style="zoom:50%;" /> |
  | 5          | <img src="https://i.loli.net/2021/09/24/2kAXruFhxRiZtPw.png" alt="5" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/VNr8bFtpBJihqKf.png" alt="5" style="zoom:50%;" /> |
  | 6          | <img src="https://i.loli.net/2021/09/24/DiF6Ot9Ll1RH3SN.png" alt="6" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/ql2pPo46mUCxiH3.png" alt="6" style="zoom:50%;" /> |
  | 7          | <img src="https://i.loli.net/2021/09/24/GtVAuTkOJ31hR6K.png" alt="7" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/gSXe7r1TQdZw5cJ.png" alt="7" style="zoom:50%;" /> |
  | 8          | <img src="https://i.loli.net/2021/09/24/nbQyuhfcHKFYOse.png" alt="8" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/yRcC2Nm5wHWZALs.png" alt="8" style="zoom:50%;" /> |
  | 9          | <img src="https://i.loli.net/2021/09/24/gxbPqHwi9Tztmjn.png" alt="9" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/OBsGXmxfIVUPT3b.png" alt="9" style="zoom:50%;" /> |
  | 10         | <img src="https://i.loli.net/2021/09/24/OJHGzXUQDP5YAwp.png" alt="10" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/liHDnevSkB4baRy.png" alt="10" style="zoom:50%;" /> |
  | 11         | <img src="https://i.loli.net/2021/09/24/dvfm5jXOigShGUe.png" alt="11" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/sU7NO64R1ryaAu8.png" alt="11" style="zoom:50%;" /> |
  | 12         | <img src="https://i.loli.net/2021/09/24/RFUXy5gh1ciADs6.png" alt="12" style="zoom:50%;" /> | <img src="https://i.loli.net/2021/09/24/DgQ3hnOIdUHbRJE.png" alt="12" style="zoom:50%;" /> |
  