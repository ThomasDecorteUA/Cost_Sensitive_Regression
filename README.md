# Interpretable Cost-Sensitive Regression through One-Step Boosting (OSB)

### Authors: Thomas Decorte, Tim Verdonck, Jakob Raymaekers

## Introduction
The Interpretable Cost-Sensitive Regression through One-Step Boosting, the OSB algorithm, is a post-hoc cost-sensitive regression method to 
account for an asymmetric cost structure in regression problems. In most practical prediction problems, 
the different types of prediction errors are not equally costly. These regressions are typically characterized by an asymmetric
cost structure, where over- and underpredictions of a similar magnitude face vastly different costs.
In the paper below and the code here, we present a one-step boosting method for cost-sensitive regression. The proposed
methodology leverages a secondary learner to incorporate cost-sensitivity into an already trained
cost-insensitive regression model. The secondary learner is defined as a linear function of certain
variables deemed interesting for cost-sensitivity. These variables do not necessarily need to be the
same as in the already trained model. An efficient optimization algorithm is achieved through iter-
atively reweighted least squares using the asymmetric cost function. The obtained results become
interpretable through bootstrapping, enabling decision makers to distinguish important variables
for cost-sensitivity as well as facilitating statistical inference. Three different cost functions are implemented to 
showcase the algorithm. 

For more information see: https://www.sciencedirect.com/science/article/abs/pii/S0167923623000994

## Dependencies

The model was implemented in Python 3.9. The following packages are needed for running the model:
- numpy==1.23.3
- pandas==1.5.0
- scikit-learn==1.1.2
- scipy==1.9.1
- sklearn==0.0
- statsmodels==0.13.2

To run the below example, also the following is needed:
- lightgbm==3.3.2

```
Options :
  --type	           # Type of cost function applied to dataset. Can be: LinLin (linlin), QuadQuad (quadquad), LinEx (lin_ex) or ExLin (ex_lin) 
  --a	                   # Parameter for cost penalizations of overpredictions
  --b                      # Parameter for cost penalizations of underpredictions
```

Outputs:
   - Initial average misprediction cost
   - Coefficients of the boosting step found by iteratively least squares based on the given input
   - Post-hoc average misprediction cost

## Datasets
The datasets showcased in the paper are included in this repository, namely  Abalone, Bank (8FM), House (8L), and KC House. The first three of these datasets are made available on the DELVE (https://www.cs.toronto.edu/~delve/data/datasets.html) (Data for Evaluating Learning in Valid Experiments) repository of the University of Toronto or on the UCI repository (https://archive.ics.uci.edu/ml/datasets.php). The last dataset, KC House, is made available by the  Center for Spatial Data Science (https://geodacenter.github.io/data-and-lab//KingCounty-HouseSales2015/) at the University of Chicago.

## Example usage

To execute an example of the procedure based on the KC House dataset run the following line:
```
 python example.py --b=1 --a=10 --type="linlin"
```


