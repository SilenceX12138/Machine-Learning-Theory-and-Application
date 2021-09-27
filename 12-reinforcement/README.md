# Reinforcement Learning

## Basic Concept

* Categorical Distribution: Linear mapping of every probability to make them sum up to 1
  * theory: https://en.wikipedia.org/wiki/Categorical_distribution
  * implement: https://pytorch.org/docs/stable/distributions.html#categorical
* `matplotlib.pyplot.gcf()`: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gcf.html

## Experiment

|                 Method Name                  |                        Total Rewards                         |                        Final Rewards                         | Test Reward |
| :------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------: |
|                  Version 0                   |   ![1](https://i.loli.net/2021/09/27/Xpr5nTLty7R6Vlf.png)    |   ![2](https://i.loli.net/2021/09/27/pbIgBkO6EFGyReL.png)    |  $-12.23$   |
|                  Version 2                   | ![download](https://i.loli.net/2021/09/27/qM4NEbA3GUwSs9I.png) | ![download (1)](https://i.loli.net/2021/09/27/F5O9EtLZuIP8oVf.png) |   $10.88$   |
| Version 4 (Advantage Actor-Critic, 1000 epoch) | ![download](https://i.loli.net/2021/09/27/uik8NaoUxzspTwG.png) | ![download (2)](https://i.loli.net/2021/09/27/NmL2XDKQizuodC3.png) |  $-22.57$   |

* Actor Critic converge much **slower** than other two methods.
