from torch import nn


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args):
        for i in args:
            assert isinstance(i, (tuple, list)), "all args must be either tuple or list, got %s" % type(i)
            # we could check for equal lengths here as well but we really shouldn't overdo it with checks because
            # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = [1] * len(args[0])
        else:
            weights = self.weight_factors

        # 給 per-component logging 用：如果內層 loss 是 Compound_loss，會在每次 call 後
        # 寫進 self.loss.last_components；我們把各 scale 的值依 deep-supervision weight 加總起來
        self.last_components = None

        # we initialize the loss like this instead of 0 to ensure it sits on the correct device, not sure if that's
        # really necessary
        l = weights[0] * self.loss(*[j[0] for j in args])
        # 抓取 scale 0 的 per-component 值
        inner_lc = getattr(self.loss, 'last_components', None)
        if inner_lc:
            self.last_components = {n: weights[0] * v for n, v in inner_lc.items()}
        for i, inputs in enumerate(zip(*args)):
            if i == 0:
                continue
            l += weights[i] * self.loss(*inputs)
            inner_lc = getattr(self.loss, 'last_components', None)
            if inner_lc and self.last_components is not None:
                for n, v in inner_lc.items():
                    self.last_components[n] = self.last_components.get(n, 0.0) + weights[i] * v
        return l