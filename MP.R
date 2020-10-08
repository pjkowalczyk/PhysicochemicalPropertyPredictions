library(rcdk)
library(tidyverse)
library(magrittr)
library(purrr)
library(stringr)
library(caret)
library(corrplot)
library(ggplot2)
library(ggthemes)

# read data

## training data
train <-
  read.csv('cache/TR_MP_6486_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(MP, everything()) %>%
  na.omit()

X_train <- train %>%
  select(-MP)
y_train <- train %>%
  select(MP) %>%
  data.frame()

## test data
test <-
  read.csv('cache/TST_MP_2167_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(MP, everything()) %>%
  na.omit()

X_test <- test %>%
  select(-MP)
y_test <- test %>%
  select(MP) %>%
  data.frame()

#####
TRAIN <- train %>%
  mutate(set = 'train')
TEST <- test %>%
  mutate(set = 'test')
MP <- rbind(TRAIN, TEST)
MP_train_test <- ggplot(MP, aes(MP, stat(density), colour = set)) +
  geom_freqpoly(binwidth = 10, size = 1) +
  scale_color_manual(values = c('#EB6B4A', '#0B3087')) +
  theme(legend.position="none")
MP_train_test

# curate data

## near-zero variance descriptors

nzv <- nearZeroVar(X_train, freqCut = 100/0)
X_train <- X_train[ , -nzv]
### and
X_test <- X_test[ , -nzv]

## highly correlated descriptors

correlations <- cor(X_train)
corrplot::corrplot(correlations, order = 'hclust')
highCorr <- findCorrelation(correlations, cutoff = 0.85)
X_train <- X_train[ , -highCorr]
### and
X_test <- X_test[ , -highCorr]

correlations <- cor(X_train)
corrplot::corrplot(correlations, order = 'hclust')

## linear combinations

comboInfo <- findLinearCombos(X_train) # returns NULL
# X_train <- X_train[ , -comboInfo$remove]
# ### and
# X_test <- X_test[ , -nzv]

## center & scale descriptors

preProcValues <- preProcess(X_train, method = c("center", "scale"))

X_trainTransformed <- predict(preProcValues, X_train)
### and
X_testTransformed <- predict(preProcValues, X_test)

### PCA

pca <- preProcess(X_trainTransformed, method = c('pca'))
X_train_pca <- predict(pca, X_trainTransformed)
X_test_pca <- predict(pca, X_testTransformed)

train_pca <- X_train_pca %>%
  select(PC1, PC2) %>%
  mutate(dataset = 'train')
test_pca <- X_test_pca %>%
  select(PC1, PC2) %>%
  mutate(dataset = 'test')
pcaPts <- rbind(train_pca, test_pca)

MP_PC <-
  ggplot(pcaPts, aes(PC1, PC2)) +
  geom_point(aes(colour = factor(dataset), shape = factor(dataset))) +
  labs(title = 'MP') +
  theme(legend.position="none")
MP_PC

# models

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  repeats = 5)

set.seed(350)

## multiple linear regression

trainSet <- cbind(y_train, X_trainTransformed)

mlr <- train(MP ~ .,
             data = trainSet,
             method = 'lm',
             trControl = fitControl)

y_predict <- predict(mlr, newdata = X_testTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_test, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

MP_mlr <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  # coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method = 'lm') +
  labs(title = 'Multiple Linear Regression') +
       # subtitle = 'Multiple Linear Regression\n test data') +
  ggthemes::theme_tufte()
MP_mlr <- MP_mlr + geom_abline(intercept = 0,
                               slope = 1,
                               colour = 'red')
MP_mlr

y_predict <- predict(mlr, newdata = X_trainTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

mlrPR <- postResample(pred = y_predict, obs = X_trainTransformed)
rmse_train = c(mlrPR[1])
r2_train = c(mlrPR[2])

data2plot <- cbind(y_train, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

p <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method='lm') +
  labs(title = 'MP',
       subtitle = 'Multiple Linear Regression\n training data') +
  ggthemes::theme_tufte()
p <- p + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
p

## partial least squares

plsModel <- train(
  MP ~ .,
  data = trainSet,
  method = 'pls',
  tuneLength = 20,
  trControl = fitControl
)

y_predict <- predict(plsModel, newdata = X_testTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_test, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

MP_pls <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  # coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method = 'lm') +
  labs(title = 'Partial Least Squares') +
       # subtitle = 'Partial Least Squares\n test data') +
  ggthemes::theme_tufte()
MP_pls <- MP_pls + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
MP_pls

y_predict <- predict(plsModel, newdata = X_trainTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_train, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

p <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method='lm') +
  labs(title = 'MP',
       subtitle = 'Partial Least Squares\n training data') +
  ggthemes::theme_tufte()
p <- p + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
p

## support vector machines

svmModel <- train(
  MP ~ .,
  data = trainSet,
  method = 'svmRadial',
  # tuneLength = 14,
  trControl = fitControl
)

y_predict <- predict(svmModel, newdata = X_testTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_test, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

MP_svm <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  # coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method = 'lm') +
  labs(title = 'Support Vector Machines') +
       # subtitle = 'Support Vector Machines\n test data') +
  ggthemes::theme_tufte()
MP_svm <- MP_svm + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
MP_svm

y_predict <- predict(svmModel, newdata = X_trainTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_train, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

p <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method='lm') +
  labs(title = 'MP',
       subtitle = 'Support Vector Machines\n training data') +
  ggthemes::theme_tufte()
p <- p + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
p

## k-nearest neighbors

knnModel <- train(
  MP ~ .,
  data = trainSet,
  method = 'knn',
  tuneGrid = data.frame(.k = 1:20),
  trControl = fitControl
)

y_predict <- predict(knnModel, newdata = X_testTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_test, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

MP_kNN <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  # coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method = 'lm') +
  labs(title = 'K-Nearest Neighbors') +
       # subtitle = 'K-Nearest Neighbors\n test data') +
  ggthemes::theme_tufte()
MP_kNN <- MP_kNN + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
MP_kNN

y_predict <- predict(knnModel, newdata = X_trainTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_train, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

p <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method='lm') +
  labs(title = 'MP',
       subtitle = 'K-Nearest Neighbors\n training data') +
  ggthemes::theme_tufte()
p <- p + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
p

## boosted trees

gbmGrid <- expand.grid(
  .interaction.depth = seq(1, 7, by = 2),
  .n.trees = seq(100, 1000, by = 50),
  .shrinkage = c(0.001, 0.1),
  .n.minobsinnode = 3
)

treeModel <- train(
  MP ~ .,
  data = trainSet,
  method = 'gbm',
  tuneGrid = gbmGrid,
  verbose = FALSE
)

y_predict <- predict(treeModel, newdata = X_testTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_test, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

MP_gbm <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  # coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method = 'lm') +
  labs(title = 'Boosted Trees') +
       # subtitle = 'Boosted Trees\n test data') +
  ggthemes::theme_tufte()
MP_gbm <- MP_gbm + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
MP_gbm

y_predict <- predict(treeModel, newdata = X_trainTransformed) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_train, y_predict)

summary(lm(Predicted ~ MP, data = data2plot))

p <-
  ggplot(data2plot, aes(MP, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method='lm') +
  labs(title = 'MP',
       subtitle = 'Boosted Trees\n training data') +
  ggthemes::theme_tufte()
p <- p + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
p

alles_plot <-
  gridExtra::grid.arrange(MP_train_test, MP_mlr, MP_pls, MP_svm, MP_kNN, MP_gbm, nrow = 3)
ggsave('EcoTox/MP_plots.png', alles_plot)

