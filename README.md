# Learning to Select SAT Encodings - Data and Code for Experiments

This repository contains code and results for the experiments supporting the
paper "Learning to Select SAT Encodings for Psudo-Boolean and Linear Integer
Constraints" by Felix Ulrich-Oltean, Peter Nightingale and James Walker

## Contents
- `constraint-models.tgz`: the constraint problems modelled in Essence Prime
- `savilerow-1.9.1-extract-features-XXX.tgz`: a modified version of the [Savile
  Row](https://savilerow.cs.st-andrews.ac.uk/) constraint modelling tool, based
  on realease 1.9.1 but including feature extraction and the full set of 9
  pb/li sat encodings.  We used Savile Row to solve all the problems in our
  corpus using the different encodings, as well as to help us extract features.
- `scripts/`: the code for the machine learning task.  The main work is done in
  `fullcycle.py`.
- `setups-and-results/`: the configuration details for the various ML setups we
  tried and the accompanying results
- `analysis/`: code for summarising and plotting
- `appendix-picat-trial`: models and instances for the mini experiment comparing
  picat's log encoding with Savile Row's

## Software Requirements
- a Java runtime environment (we used version 13)
- Python (we used 3.7) and at least the following packages (a full list of
  packages in our conda environment is given in
  [`conda-packages.txt`](./conda-packages.txt):
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - xgboost
  - ruamel-yaml
  - pyyaml
- a SAT solver - we used Kissat
  [sc2021-sweep](https://github.com/arminbiere/kissat/tree/sc2021-sweep)

Optional
- AutoFolio, for comparison - we used
  https://github.com/automl/AutoFolio/commit/60a38a485e832b4e9bd2d06cbe7e1aecc994bb32
  which extended release 2.1.2 with an API allowing predictions from CSV files
  of features and labels

## Questions?
Please get in touch with the authors (or open an "Issue" in github) if you have
any questions about this data, code, or work in general.
