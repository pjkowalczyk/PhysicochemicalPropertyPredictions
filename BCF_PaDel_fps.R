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
  read.csv('cache/TR_BCF_469.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(ChemID, LogBCF)
trainFP <-
  read.csv('data/BCF/trainFP.csv',
           header = TRUE,
           stringsAsFactors = FALSE)
TRAIN <- merge(train, trainFP, by.x = 'ChemID', by.y = 'Name')

## test data
test <-
  read.csv('cache/TST_BCF_157.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(ChemID, LogBCF)
testFP <-
  read.csv('data/BCF/testFP.csv',
           header = TRUE,
           stringsAsFactors = FALSE)
TEST <- merge(test, testFP, by.x = 'ChemID', by.y = 'Name')

alles <- rbind(TRAIN, TEST)

rm(TEST, test, testFP, TRAIN, train, trainFP)

include <- createDataPartition(alles$LogBCF, p = 0.8, list = FALSE, groups = 10)

train <- alles[include, ]
X_train <- train %>%
  select(-ChemID, -LogBCF)
y_train <- train %>%
  select(LogBCF)
test <- alles[-include, ]
X_test <- test %>%
  select(-ChemID, -LogBCF)
y_test <- test %>%
  select(LogBCF)

summary(train$LogBCF)
summary(test$LogBCF)

# curate data

## near-zero variance descriptors

nzv <- nearZeroVar(X_train, freqCut = 98/2)
X_train <- X_train[ , -nzv]
### and
X_test <- X_test[ , -nzv]

## highly correlated descriptors

correlations <- cor(X_train)
# corrplot::corrplot(correlations, order = 'hclust')
highCorr <- findCorrelation(correlations, cutoff = 0.85)
X_train <- X_train[ , -highCorr]
### and
X_test <- X_test[ , -highCorr]

# correlations <- cor(X_train)
# corrplot::corrplot(correlations, order = 'hclust')

## linear combinations

# comboInfo <- findLinearCombos(X_train) # returns NULL
# X_train <- X_train[ , -comboInfo$remove]
# ### and
# X_test <- X_test[ , -nzv]

### multiple linear regression

trainData <- cbind(y_train, X_train)
testData <- cbind(y_test, X_test)
allData <- rbind(trainData, testData)

X <- testData %>%
  select(-LogBCF)
y <- testData %>%
  select(LogBCF)

BCF_MLR <- lm(LogBCF ~ ., data = trainData)

X_test <- testData %>%
  select(-LogBCF)
y_test <- testData %>%
  select(LogBCF)

Pred_BCF_MLR <- predict(BCF_MLR, X_test) %>%
  data.frame()

qaz <- cbind(Pred_BCF_MLR, y_test)
colnames(qaz) <- c('pred', 'obs')

test_lm <- lm(pred ~ obs, data = qaz)

plot(qaz$pred, qaz$obs)

# models

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  repeats = 5)

set.seed(350)

## multiple linear regression

trainSet <- cbind(y_train, X_train)

mlr <- train(LogBCF ~ .,
             data = trainSet,
             method = 'lm',
             trControl = fitControl)

y_predict <- predict(mlr, newdata = X_test) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

data2plot <- cbind(y_test, y_predict)

summary(lm(Predicted ~ LogBCF, data = data2plot))

p <-
  ggplot(data2plot, aes(LogBCF, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method = 'lm') +
  labs(title = 'LogBCF',
       subtitle = 'Multiple Linear Regression\n test data') +
  ggthemes::theme_tufte()
p <- p + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
p

y_predict <- predict(mlr, newdata = X_train) %>%
  data.frame()
colnames(y_predict) <- c('Predicted')

mlrPR <- postResample(pred = y_predict, obs = X_train)
rmse_train = c(mlrPR[1])
r2_train = c(mlrPR[2])

data2plot <- cbind(y_train, y_predict)

summary(lm(Predicted ~ LogBCF, data = data2plot))

p <-
  ggplot(data2plot, aes(LogBCF, Predicted)) +
  geom_point(colour = "blue", size = 2) +
  coord_equal() +
  # xlim(c(0, 3.5)) + ylim(c(0, 3.5)) +
  geom_smooth(method='lm') +
  labs(title = 'LogBCF',
       subtitle = 'Multiple Linear Regression\n training data') +
  ggthemes::theme_tufte()
p <- p + geom_abline(intercept = 0,
                     slope = 1,
                     colour = 'red')
p

