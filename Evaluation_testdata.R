library(tidyverse)
library(pROC)

setwd('') #INPUT working directory

X_train1 <- #INPUT training data for model1
X_train2 <- #INPUT training data for model2

X_test1 <- #INPUT test data for model1
X_test2 <- #INPUT test data for model2
  
y_train <- #INPUT outcome for training data
y_test <- #INPUT outcome for test data
  
# Create two logistic regression models from training data
model1 <- glm(unlist(y_train)~.,family=binomial(link='logit'),data=X_train1)
model2 <- glm(unlist(y_train)~.,family=binomial(link='logit'),data=X_train2)

# Predict logistic regression models from test data
pred1 <- predict(model1, newdata=X_test1, type="response")
pred2 <- predict(model2, newdata=X_test2, type="response")

# Evaluate AUC and confidence intervals for the two models
auc1 <- ci.auc(unlist(y_test), pred1)
auc2 <- ci.auc(unlist(y_test), pred2)

# Significance test between the two ROC curves 
roc1 <- roc(unlist(y_test), pred1)
roc2 <- roc(unlist(y_test), pred2)
roc_test <- roc.test(roc1, roc2, method='bootstrap',alternative='less')
pval <- roc_test$p.value

###############################################################################

pvalues <- # create list of pvalues 

# adjust list of p-values for multiple comparisons by Benjamini–Hochberg method
pvals_adjust <- p.adjust(pvalues, method = 'BH')

