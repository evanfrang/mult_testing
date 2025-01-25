# Multiple Testing Problem
Evan Frangipane

- [Problem Statement](#problem-statement)
- [Outlining the Simulation](#outlining-the-simulation)
- [Discrepancy in FWER](#discrepancy-in-fwer)
- [Correcting Significance](#correcting-significance)

## Problem Statement

The main idea of the multiple testing problem is the more statistical
tests we perform during an analysis, the higher our false positive rate
(Type I Error), if left uncorrected. Imagine we choose our confidence
level to be
![95\\](https://latex.codecogs.com/svg.latex?95%5C%25 "95\%"),
essentially we are choosing our false positive rate to be
![5\\](https://latex.codecogs.com/svg.latex?5%5C%25 "5\%") for one test.
If we test again, the probability of at least one false positive is
![1 - (0.95 \* 0.95)](https://latex.codecogs.com/svg.latex?1%20-%20%280.95%20%2A%200.95%29 "1 - (0.95 * 0.95)").
If we continue testing until
![N](https://latex.codecogs.com/svg.latex?N "N"), we can rewrite the
false positive probablity (at least one) as
![1 - (1-0.05)^N](https://latex.codecogs.com/svg.latex?1%20-%20%281-0.05%29%5EN "1 - (1-0.05)^N").
This probability is called the family-wise error rate (FWER). As
![N](https://latex.codecogs.com/svg.latex?N "N") increases, the FWER
increases to probability of
![1](https://latex.codecogs.com/svg.latex?1 "1"). If we allow this
problem to get out of hand, we could be making false inferences.

## Outlining the Simulation

For our simulation we will be flipping fair coins. We are doing a
two-tailed test to see if the number of heads is significantly different
from the number of tails. Some parameters:

- ![M](https://latex.codecogs.com/svg.latex?M "M") - number of coins
  being flipped in each test
- ![\alpha = 0.05](https://latex.codecogs.com/svg.latex?%5Calpha%20%3D%200.05 "\alpha = 0.05") -
  significance for each test
- ![N](https://latex.codecogs.com/svg.latex?N "N") - number of tests
  performed
- ![NN = 10000](https://latex.codecogs.com/svg.latex?NN%20%3D%2010000 "NN = 10000") -
  number of repetitions of each analysis

So, the total number of coins flipped in each analysis is
![M \* N \* NN](https://latex.codecogs.com/svg.latex?M%20%2A%20N%20%2A%20NN "M * N * NN").
We choose
![M = \\18, 51, 120\\](https://latex.codecogs.com/svg.latex?M%20%3D%20%5C%7B18%2C%2051%2C%20120%5C%7D "M = \{18, 51, 120\}"),
and
![N \in \[1, 1000\]](https://latex.codecogs.com/svg.latex?N%20%5Cin%20%5B1%2C%201000%5D "N \in [1, 1000]")
for the following plots.

<details class="code-fold">
<summary>Code</summary>

``` python
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import binom
from matplotlib import rcParams
import pickle

plt.rc('text', usetex=True)
plt.rc('axes', linewidth=2)
rcParams['font.family'] = 'serif'
plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
#rcParams['font.serif'] = ['Times New Roman']  
rcParams['font.size'] = 14 
rcParams['axes.titlesize'] = 16  
rcParams['axes.labelsize'] = 14  
rcParams['legend.fontsize'] = 12  
rcParams['xtick.labelsize'] = 12  
rcParams['ytick.labelsize'] = 12  
plt.style.use('bmh')

with open('fwer.pkl', 'rb') as f:
    fwer_1, fwer_2, fwer_3, M_list, fwer_b1, fwer_b2, fwer_b3, \
    fwer_N, fwer_bound, fwer_bound_N, fwer_bound_b, Ms, \
    pvs, index = pickle.load(f)
```

</details>

We plot the results of our analyses in
<a href="#fig-fwer" class="quarto-xref">Figure 1</a>. The false positive
rate (FWER) increases toward
![1](https://latex.codecogs.com/svg.latex?1 "1") with
![N](https://latex.codecogs.com/svg.latex?N "N"). The three choices of
![M](https://latex.codecogs.com/svg.latex?M "M") are plotted along with
the analytical curve. There is a discrepancy between the analytical
curve and the numerical simulations that is smallest for
![M = 51](https://latex.codecogs.com/svg.latex?M%20%3D%2051 "M = 51").
We will return to this in the next section. The relevant feature is the
monotonic increase in Type I Error.

<details class="code-fold">
<summary>Code</summary>

``` python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fwer_bound_N, fwer_bound, label=r'$1 - (1 - \alpha)^N$', color='navy')
ax.plot(fwer_N, fwer_1, 'o',label=f'M={M_list[0]}', \
    color='darkred', markersize=6, alpha=0.9)
ax.plot(fwer_N, fwer_2, '^',label=f'M={M_list[1]}', \
    color='darkgreen', markersize=6, alpha=0.9)
ax.plot(fwer_N, fwer_3, 's',label=f'M={M_list[2]}', \
    color='teal', markersize=6, alpha=0.9)

ax.set_xlabel('Number of Tests (N)')
ax.set_ylabel('FWER')
ax.set_xscale('log')

ax.legend(loc='lower right', frameon=True, shadow=True, borderpad=1)
ax.grid(which='both', linestyle='-', linewidth=0.8, color='gray', alpha=0.7)
#plt.title('Family-Wise Error Rate vs. Number of Tests')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()
```

</details>

<div id="fig-fwer">

![](README_files/figure-commonmark/fig-fwer-output-1.png)

Figure 1: Uncorrected FWER

</div>

## Discrepancy in FWER

Our ![\alpha](https://latex.codecogs.com/svg.latex?%5Calpha "\alpha")
significance was chosen to be
![0.05](https://latex.codecogs.com/svg.latex?0.05 "0.05") for this
analysis. However, when we are dealing with coin flips we are using
discrete data and not continuous. This has consequences for what
constitutes a significant result for flipping
![M](https://latex.codecogs.com/svg.latex?M "M") coins. For a
significant result we need the p-value of the number of heads (or tails
hence two-tail test) to be the largest number less than
![\alpha = 0.05](https://latex.codecogs.com/svg.latex?%5Calpha%20%3D%200.05 "\alpha = 0.05").
We call this number of heads the critical number of heads. However, due
to discrete data, the critical p-value can vary greatly rather than
being exactly ![0.05](https://latex.codecogs.com/svg.latex?0.05 "0.05").
One would expect that as the number of coin flips increases, the
critical p-value should approach
![0.05](https://latex.codecogs.com/svg.latex?0.05 "0.05"). The intuition
being as we increase the number of coins flipped we are filling in the
discrete data set and approaching continuum. We plot the critical
p-value for increasing ![M](https://latex.codecogs.com/svg.latex?M "M")
coin flips in <a href="#fig-crit" class="quarto-xref">Figure 2</a>.
Additionally, we also plot the three
![M](https://latex.codecogs.com/svg.latex?M "M") values from the
previous simulation and they show the same hierarchy as seen in
<a href="#fig-fwer" class="quarto-xref">Figure 1</a>. The closer the
critical p-value is to
![0.05](https://latex.codecogs.com/svg.latex?0.05 "0.05"), the closer
the FWER is to
![1 - (1 - \alpha)^N](https://latex.codecogs.com/svg.latex?1%20-%20%281%20-%20%5Calpha%29%5EN "1 - (1 - \alpha)^N").

<details class="code-fold">
<summary>Code</summary>

``` python
colors = ['red', 'green', 'cyan']
labels = [f'M={M_list[0]}', f'M={M_list[1]}', f'M={M_list[2]}']
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Ms, pvs, color='navy', label='Critical p-Value', alpha=0.8)
ax.axhline(0.05, color='black', label=f'$\\alpha = 0.05$')
plt.plot([Ms[index[0]]], [pvs[index[0]]], 'o', label=labels[0], \
    color='darkred', markersize=8, zorder=2)
plt.plot([Ms[index[1]]], [pvs[index[1]]], '^', label=labels[1], \
    color='darkgreen', markersize=8, zorder=2)
plt.plot([Ms[index[2]]], [pvs[index[2]]], 's', label=labels[2], \
    color='teal', markersize=8, zorder=2)
ax.set_xscale('log')
ax.set_xlabel('M')
ax.set_ylabel('p-value')
#plt.title('p-value of Critical Integer $(p_{\\text{crit}}< \\alpha)$')
ax.legend(loc='lower right', frameon=True, shadow=True, borderpad=1)
ax.grid(which='both', linestyle='-', linewidth=0.8, color='gray', alpha=0.7)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout() 
plt.show()
```

</details>

<div id="fig-crit">

![](README_files/figure-commonmark/fig-crit-output-1.png)

Figure 2: Critical p-value

</div>

## Correcting Significance

A simple solution to this problem is to scale our choice of
![\alpha](https://latex.codecogs.com/svg.latex?%5Calpha "\alpha") for
each test by the number of tests. The simplest correction is the
Bonferroni correction, which is simply
![\alpha \rightarrow \alpha/N](https://latex.codecogs.com/svg.latex?%5Calpha%20%5Crightarrow%20%5Calpha%2FN "\alpha \rightarrow \alpha/N").
This comes from taking the Taylor Expansion of the FWER equation with
small parameter
![\alpha](https://latex.codecogs.com/svg.latex?%5Calpha "\alpha").

![1 - (1 - \alpha)^N \rightarrow 1 - (1 - N\*\alpha + \mathcal{O}(\alpha^2)) = N\*\alpha + \mathcal{O}(\alpha^2)](https://latex.codecogs.com/svg.latex?1%20-%20%281%20-%20%5Calpha%29%5EN%20%5Crightarrow%201%20-%20%281%20-%20N%2A%5Calpha%20%2B%20%5Cmathcal%7BO%7D%28%5Calpha%5E2%29%29%20%3D%20N%2A%5Calpha%20%2B%20%5Cmathcal%7BO%7D%28%5Calpha%5E2%29 "1 - (1 - \alpha)^N \rightarrow 1 - (1 - N*\alpha + \mathcal{O}(\alpha^2)) = N*\alpha + \mathcal{O}(\alpha^2)")

Given this expansion, a natural redefinition of
![\alpha](https://latex.codecogs.com/svg.latex?%5Calpha "\alpha") is
![\alpha = \alpha / N](https://latex.codecogs.com/svg.latex?%5Calpha%20%3D%20%5Calpha%20%2F%20N "\alpha = \alpha / N")
such that the first term in the expansion is our new
![\alpha = 0.05](https://latex.codecogs.com/svg.latex?%5Calpha%20%3D%200.05 "\alpha = 0.05").
This redefinition bounds the FWER to
![0.05](https://latex.codecogs.com/svg.latex?0.05 "0.05") rather than
asymptoting to ![1](https://latex.codecogs.com/svg.latex?1 "1"). Now the
entire analysis has a significance of
![0.05](https://latex.codecogs.com/svg.latex?0.05 "0.05"), while each
individual test has a smaller significance scaled by the number of
tests. Now, we redo the analysis with our new significance to confirm
that FWER is around or less than
![0.05](https://latex.codecogs.com/svg.latex?0.05 "0.05"). This can be
seen in <a href="#fig-fwer-bon" class="quarto-xref">Figure 3</a>. Notice
that there is no clear hierarchy between the three choices of
![M](https://latex.codecogs.com/svg.latex?M "M"), this can be explained
by <a href="#fig-crit" class="quarto-xref">Figure 2</a> again, where
this time because
![\alpha](https://latex.codecogs.com/svg.latex?%5Calpha "\alpha")
depends on ![N](https://latex.codecogs.com/svg.latex?N "N"), the
critical p-values will vary with
![N](https://latex.codecogs.com/svg.latex?N "N") and thus the hierarchy
will vary.

<details class="code-fold">
<summary>Code</summary>

``` python
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(fwer_bound_N, fwer_bound_b, label=r'$1 - (1 - \alpha/N)^N$', \
    color='navy', linewidth=2)
ax.plot(fwer_N, fwer_b1, 'o', label=f'M={M_list[0]} Bonferroni', \
    color='darkred', markersize=6, alpha=0.9)
ax.plot(fwer_N, fwer_b2, '^', label=f'M={M_list[1]} Bonferroni', \
    color='darkgreen', markersize=6, alpha=0.9)
ax.plot(fwer_N, fwer_b3, 's', label=f'M={M_list[2]} Bonferroni', \
    color='teal', markersize=6, alpha=0.9)

ax.set_xlabel('Number of Tests (N)')
ax.set_ylabel('FWER')
ax.set_xscale('log')

ax.legend(loc='lower left', frameon=True, shadow=True, borderpad=1)
ax.grid(which='both', linestyle='-', linewidth=0.8, color='gray', alpha=0.7)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()
```

</details>

<div id="fig-fwer-bon">

![](README_files/figure-commonmark/fig-fwer-bon-output-1.png)

Figure 3: Bonferroni Corrected FWER

</div>
