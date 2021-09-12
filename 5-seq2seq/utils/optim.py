# %% [markdown]
# ## Optimizer: Adam + lr scheduling
# Inverse square root 排程對於訓練 Transformer 時的穩定性很重要，後來也用在 RNN 上。
# 根據底下公式來更新 learning rate，前期線性增長，後期根據更新步數方根的倒數來遞減。
# $$lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$$
# code [source](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * (self.model_size**(-0.5) * min(
            step**(-0.5), step * self.warmup**(-1.5)))
