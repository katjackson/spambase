# Spam Classification
## Bayesian Models in scikit-learn

In this exercise I used scikit-learn to create and train a Bayesian classifier to discern spam from ham (non-spam emails). I used pandas to import and manage the data, which came from the from UCI Machine Learning Repository. The classifier that was trained on all of the data had an R<sup>2</sup> score of 0.78. Intent on finding a more reliable model, I tried linear regression, and a more Bayesian moedls with different combinations of features. The most reliable method involves a few of the most significant word counts and a character count for the '!' symbol. This model performed with an R<sup>2</sup> score of 0.88.

All of my findings can be viewed in [this jupyter notebook](https://github.com/katjackson/spambase/blob/master/Spambase.ipynb).
