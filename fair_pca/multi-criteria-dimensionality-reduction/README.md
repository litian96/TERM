# multiCriteriaDimReduction

This repo contains the codes of the paper Multi-Criteria Dimensionality Reduction with Applications to Fairness.

The .py files contain helper methods for:
- preprocessing the data
- standard PCA and calculating several utility criteria of fairness
- solving SDP
- using mutiplicative weight update method for some types of objective functions
- other helper methods

Each of the Jupyter notebooks shows how to apply one of the PCA strategy (fair SDP-based PCA vs standard PCA) to each of the dataset (Credit and Income data). The MW notebook implements MW (multiplicative weight update) instead of using SDP solver for the fair SDP-based PCA.

To use, we recommend opening one of the Jupyter notebooks and running through it. We recommend opening either:
- SDP_credit.ipynb
- heuristics.ipynb in 'additional-heuristics' folder

You can check the format of the data and see an example usage, and apply to your own datasets. For more details of the usage, each method (MW, fair SDP-based PCA, std_PCA, etc.) has documentation of usage in corresponding .py files, including what to expect as an input and output. The *Notations* subsection also gives a quick explanation of variable names and keywords.

### Contact 
If you have any question, feel free to reach me at [my username] [at] gmail dot com.

## Variants of Algorithms

In the 'additional-heuristics' folder, we develop MW and FW alternatives to solving SDP which is much faster in practice and apply it to a large Census data of original dimensions n=1000,2000. The runtime of these are in order of a minute for data of up to 1000 original dimensions, and about 10 minutes for up to 2000 dimensions. We recommend using these alternatives for n>=50.

We note that MW is also used for income data ('MW_credit_income_example') as an example. The MW implemented is the same as in the 'additional-heuristics' folder, but simpler as it was developed for a small dataset. However, for full functionality and tutorial of using MW and FW, we still recommend exploring 'additional-heuristics'.

## Notations
This subsection explains variables and keywords used in the code.

### Variable names:

| Variables | Description |
| --- | --- |
| n | number of original dimensions |
| k | number of groups |
| m | number of constraints in an optimization. In this context, it is the same as k |
| d | number of target dimensions (to project the data to) |
| B | a list of k data matrices. The data matrix B[i] is of size n by n, obtained by computing A^T A for a data matrix A of size m x n (m datapoints in R^n) |

### Variable names, for multiplicative weight update method:

| Variables | Description |
| --- | --- |
| eta | learning rate |
| T | number of iterations |

### Objectives
Fairness criteria can be specified for both SDP_based algorithms and multiplcaitive weight algorithm. To describe them, first we explain the utility of or the objective that each group receives upon getting a PCA solution P.

#### Objective for each group
Objective for each group of a given PCA solution P:

| Objective | Description |
| --- | --- |
| 'Var' | the variance of P on that group |
| 'Best' | the value of highest variance a group may acheieve (this is a constant depending on the data, independent of P) |
| 'Loss' | the marginal of variance acheived by P from the best one, i.e. Var - Best. The marginal is negative |

These objectives for any solution P computed from our algorithms are recorded in the .csv files in data folder, followed by the group index starting from 0, e.g. Var0 (variance of group 0) or Loss5 (loss of group 5).

#### Objectives across all groups (i.e. fairness criteria)
MM_Loss, MM_Var, NSW are three fairness criteria, corresponding to the following objectives:

| Fairness Criteria | Description |
| --- | --- |
| MM_Loss | maximizing minimum mariginal loss across groups (if the number is positive), or minimizing the maximum marginal different of variance across groups (if the number is negative), i.e. trying to maximize min_{all groups i} Loss_i |
| MM_Var | maximizing minimum variance across all groups, i.e. maximizing min_{all groups i} Var_i |
| NSW | maximizing Nash social welfare of variances, which is the geometric mean of the variance across groups, i.e. maximizing (product_{all groups i} Var_i)^(1/d) |
  
## Note

This work expends the prior work "The Price of Fair PCA: One Extra Dimension" at https://github.com/uthaipon/Fair-PCA. The algorithms in this repo are applicable to more general settings. This repo is in Python (Jypyter notebook) and the previous version (Fair PCA) was implemented in MATLAB.
