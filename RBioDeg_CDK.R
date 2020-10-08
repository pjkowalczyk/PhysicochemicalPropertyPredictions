library(rcdk)
library(tidyverse)
library(magrittr)
library(purrr)
library(stringr)
library(caret)
library(corrplot)
library(ggplot2)
library(ggthemes)
library(randomForest)
library(caret)
library(e1071)

# features

descNames <-
  c(
    'org.openscience.cdk.qsar.descriptors.molecular.WeightDescriptor',
    'org.openscience.cdk.qsar.descriptors.molecular.ALOGPDescriptor',
    'org.openscience.cdk.qsar.descriptors.molecular.HBondAcceptorCountDescriptor',
    'org.openscience.cdk.qsar.descriptors.molecular.HBondDonorCountDescriptor',
    'org.openscience.cdk.qsar.descriptors.molecular.TPSADescriptor',
    'org.openscience.cdk.qsar.descriptors.molecular.RotatableBondsCountDescriptor',
    'org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorCharge',
    'org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorMass',
    'org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorPolarizability',
    'org.openscience.cdk.qsar.descriptors.molecular.BCUTDescriptor'
  )

# img <- view.image.2d(parse.smiles("B([C@H](CC(C)C)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O")[[1]])
# plot(1:10, 1:10, pch='')
# rasterImage(img, 1,3, 4,10)

# read data

train <- load.molecules('data/TR_RBioDeg_1197.sdf', aromaticity = TRUE, typing = TRUE)
records <- length(train)
train_df <- data.frame(
  NAME = character(records),
  SMILES = character(records),
  Ready_Biodeg = numeric(records),
  stringsAsFactors = FALSE
)
for (i in 1:records) {
  train_df$NAME[[i]] <- get.properties(train[[i]])$NAME
  train_df$SMILES[[i]] <- get.properties(train[[i]])$SMILES
  train_df$Ready_Biodeg[[i]] <- get.properties(train[[i]])$Ready_Biodeg
}

smiles <- as.character(train_df$SMILES)

badSmiles <- matrix()
for (i in 1:length(smiles)) {
  qw <- try(mols <- parse.smiles(smiles[i]), silent = TRUE)
  if(class(qw) == 'try-error') badSmiles <- append(badSmiles, i)
}
# badSmiles <- badSmiles[-1]
# train_df <- train_df[-badSmiles, ]

mols <- parse.smiles(smiles)
train_descs <- eval.desc(mols, descNames)

test <- load.molecules('data/TST_RBioDeg_411.sdf', aromaticity = TRUE, typing = TRUE)
records <- length(test)
test_df <- data.frame(
  NAME = character(records),
  SMILES = character(records),
  Ready_Biodeg = numeric(records),
  stringsAsFactors = FALSE
)
for (i in 1:records) {
  test_df$NAME[[i]] <- get.properties(test[[i]])$NAME
  test_df$SMILES[[i]] <- get.properties(test[[i]])$SMILES
  test_df$Ready_Biodeg[[i]] <- get.properties(test[[i]])$Ready_Biodeg
}

smiles <- as.character(test_df$SMILES)

badSmiles <- matrix()
for (i in 1:length(smiles)) {
  qw <- try(mols <- parse.smiles(smiles[i]), silent = TRUE)
  if(class(qw) == 'try-error') badSmiles <- append(badSmiles, i)
}
# badSmiles <- badSmiles[-1]
# train_df <- train_df[-badSmiles, ]

mols <- parse.smiles(smiles)
test_descs <- eval.desc(mols, descNames)

## training data
X_train <- train_descs
y_train <- train_df %>%
  select(Ready_Biodeg) %>%
  data.frame()
TRAIN <- cbind(y_train, X_train) %>%
  na.omit()
X_train <- TRAIN %>%
  select(-Ready_Biodeg)
y_train <- TRAIN %>%
  select(Ready_Biodeg) %>%
  data.frame()
y_train$Ready_Biodeg <- ifelse(y_train$Ready_Biodeg > 0.5, 'RB', 'NRB')
y_train$Ready_Biodeg <- as.factor(y_train$Ready_Biodeg)

## test data
X_test <- test_descs
y_test <- test_df %>%
  select(Ready_Biodeg) %>%
  data.frame()
TEST <- cbind(y_test, X_test) %>%
  na.omit()
X_test <- TEST %>%
  select(-Ready_Biodeg)
y_test <- TEST %>%
  select(Ready_Biodeg) %>%
  data.frame()
y_test$Ready_Biodeg <- ifelse(y_test$Ready_Biodeg > 0.5, 'RB', 'NRB')
y_test$Ready_Biodeg <- as.factor(y_test$Ready_Biodeg)

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
X_train <- X_train[ , -comboInfo$remove]
### and
X_test <- X_test[ , -comboInfo$remove]

## center & scale descriptors

preProcValues <- preProcess(X_train, method = c("center", "scale"))

X_trainTransformed <- predict(preProcValues, X_train)
### and
X_testTransformed <- predict(preProcValues, X_test)

# models

set.seed(350)

ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10,
  classProbs = TRUE)

#####...#####...#####...#####

# metric: Accuracy
metric <- "Accuracy"

tunegrid <- expand.grid(.mtry=mtry)

data2model <- cbind(y_train, X_trainTransformed)

RBioDeg_rf <- train(
  Ready_Biodeg ~ .,
  data = data2model,
  method = 'rf',
  metric = 'Accuracy',
  tuneGrid = tunegrid,
  trControl = ctrl
)

print(RBioDeg_rf)

y_predict <- predict(RBioDeg_rf, X_testTransformed) %>%
  data.frame()
colnames(y_predict) <- 'Ready_Biodeg'
y_predict$Ready_Biodeg <- as.factor(y_predict$Ready_Biodeg)

confusionMatrix(data = y_predict$Ready_Biodeg,
                reference = y_test$Ready_Biodeg,
                positive = 'NRB')

##### Model Employment #####

mols <- parse.smiles('Cn1c2nc[nH]c2c(=O)n(C)c1=O')[[1]]

M <- eval.desc(mols, descNames)

M_predict <- predict(RBioDeg_rf, M) %>%
  data.frame()

M_predict
