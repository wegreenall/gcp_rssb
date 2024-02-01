Code for the paper "Orthogonal Series Gaussian Cox Processes" as submitted
to ICML 2024.

It is based in PyTorch as its backend library. It also requires

`pandas`\
`matplotlib`\


This code depends on our proprietary libraries, `ortho` and `mercergp`, 
libraries for handling orthonormal bases and sparse Gaussian processes
respectively. Anonymised links to both of these are provided in the 
supplementary material of the paper.

Notes on replication:
- for the method by which we conduct thinning and generate Poisson process 
samples, see `data.py`

- for the empirical coverage (EC) metric, see `empirical_coverage.py`

- for base code, see `methods/gcp_ose.py`. Several classes in here are defunct.
- for the Bayesian code, see `methods/gcp_ose_bayesian.py`
- for the classification code, see `methods/gcp_ose_classifier.py`

- code for constructing the diagrams found in the paper is found in:
    - `classification_diagrams.py`
    - `classification_diagrams_2d.py`
    - `experiment_graphs.py`

which depend on classes found in `plot.py` for standardising plot behaviour.
