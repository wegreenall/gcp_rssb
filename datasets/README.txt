synth1: λ(t) = 2 * exp(- t / 15) + exp(-1. * ((t - 25) / 10) ** 2) over domain T = [0, 50]

synth2: λ(t) = 5 * sin(t ** 2) + 6 over domain T = [0, 5]

synth3: piecewise linear over domain T = [0, 100]

def piecewise_linear(X: torch.Tensor):
   idx_less_25 = [i for i in range(len(X)) if X[i] < 25]
   idx_less_50 = [i for i in range(len(X)) if 25 <= X[i] < 50]
   idx_less_75 = [i for i in range(len(X)) if 50 <= X[i] < 75]
   other_idx = [i for i in range(len(X)) if X[i] >= 75]
   return torch.cat([0.04 * X[idx_less_25] + 2, -0.08 * X[idx_less_50] + 5, 0.06 * X[idx_less_75] - 2, 0.02 * X[other_idx] + 1])

^
|
|
|_ _ _ _ might be represented more efficiently but it's used just for plots so it's not big deal i guess :)
