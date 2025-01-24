# Statistics Tinkering
Evan Frangipane

- [Multiple Testing Problem](#multiple-testing-problem)

## Multiple Testing Problem

<details class="code-fold">
<summary>Code</summary>

``` python
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import binom

random.seed(42)
P = 0.5

def fwer_upper_bound(alpha_, N_):
    return 1 - (1 - alpha_)**N_

def asymp(N_, NN_, M_, alpha_):
    results = np.random.randint(0, 2, size=(NN_, N_, M_))
    evens = np.sum(results % 2 == 0, axis=2)
    temp_binomial = binom(M_,P)
    temp_crit = int(temp_binomial.ppf(alpha_ / 2)-1.)
    false_positives = np.sum((evens <= temp_crit) 
        | (evens >= M_ - temp_crit), axis=1) > 0
    return np.mean(false_positives)

numer_tests = np.logspace(0,3,15)
low_M = 18
numer_fwer_low = [asymp(int(i), 10000, low_M, 0.05) for i in numer_tests]
med_M = 51
numer_fwer_med = [asymp(int(i), 10000, med_M, 0.05) for i in numer_tests]
high_M = 120
numer_fwer_high = [asymp(int(i), 10000, high_M, 0.05) for i in numer_tests]

numer_fwer_bonferroni_low = \
    [asymp(int(i), 10000, low_M, 0.05/i) for i in numer_tests]
numer_fwer_bonferroni_med = \
    [asymp(int(i), 10000, med_M, 0.05/i) for i in numer_tests]
numer_fwer_bonferroni_high = \
    [asymp(int(i), 10000, high_M, 0.05/i) for i in numer_tests]

ubound_tests = np.arange(1, 1000, 0.1)
ubound_fwer = 1 - (1 - 0.05)**ubound_tests

plt.plot(ubound_tests, ubound_fwer, label='FWER Estimate', color='blue')
plt.plot(numer_tests, numer_fwer_low, 'ro',label=f'FWER Numerical M={low_M}')
plt.plot(numer_tests, numer_fwer_med, 'go',label=f'FWER Numerical M={med_M}')
plt.plot(numer_tests, numer_fwer_high, 'co',label=f'FWER Numerical M={high_M}')
plt.xlabel('Number of Tests (N)')
plt.ylabel('FWER')
plt.xscale('log')
plt.title('Family-Wise Error Rate vs. Number of Tests')
plt.legend()
plt.grid()
plt.show()
```

</details>

![](README_files/figure-commonmark/cell-2-output-1.png)

fdkhfkdjsfhsd

<details class="code-fold">
<summary>Code</summary>

``` python
ubound_fwer_bonferroni = 1 - (1 - 0.05/ubound_tests)**ubound_tests

plt.plot(ubound_tests, ubound_fwer_bonferroni, label='FWER Estimate', color='blue')
plt.plot(numer_tests, numer_fwer_bonferroni_low, 'ro',label=f'FWER Numerical M={low_M} Bonferroni Correction')
plt.plot(numer_tests, numer_fwer_bonferroni_med, 'go',label=f'FWER Numerical M={med_M} Bonferroni Correction')
plt.plot(numer_tests, numer_fwer_bonferroni_high, 'co',label=f'FWER Numerical M={high_M} Bonferroni Correction')
plt.xlabel('Number of Tests (N)')
plt.ylabel('FWER')
plt.xscale('log')
plt.title('Family-Wise Error Rate vs. Number of Tests')
plt.legend()
plt.grid()
plt.show()
```

</details>

![](README_files/figure-commonmark/cell-3-output-1.png)

fdfdfdfdfdf

<details class="code-fold">
<summary>Code</summary>

``` python
Ms = np.logspace(np.log(3),np.log(500),5000).astype(int)
bins = [binom(i,P) for i in Ms]
crits = [int(i.ppf(0.05 / 2)-1.) for i in bins]
pvs = [2.*bins[i].cdf(min(Ms[i]-crits[i],crits[i])) for i in range(len(Ms))]
index = [min(range(len(Ms)), key=lambda i: abs(Ms[i]-low_M)), 
             min(range(len(Ms)), key=lambda i: abs(Ms[i]-med_M)), 
             min(range(len(Ms)), key=lambda i: abs(Ms[i]-high_M))
             ]
colors = ['red', 'green', 'cyan']
labels = [f'M={low_M}', f'M={med_M}', f'M={high_M}']
plt.figure()
plt.plot(Ms, pvs, color='blue', label='Critical p-Value')
plt.axhline(0.05, color='black', label=f'$\\alpha = 0.05$')
for i in range(len(index)):
    plt.scatter(Ms[index[i]], pvs[index[i]], color=colors[i], label=labels[i], zorder=2)
plt.xscale('log')
plt.xlabel('M')
plt.ylabel('p-value')
plt.title('p-value of Critical Integer $(p_{\\text{crit}}< \\alpha)$')
plt.legend()
plt.grid()
plt.show()
```

</details>

![](README_files/figure-commonmark/cell-4-output-1.png)
