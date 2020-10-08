library(rcdk)
library(tidyverse)
library(magrittr)
library(purrr)
library(stringr)
library(caret)
library(corrplot)
library(ggplot2)
library(ggthemes)
library(gridExtra)

# read data

## LogP

## training data
train <-
  read.csv('cache/TR_LogP_10537_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(LogP, everything()) %>%
  na.omit() %>%
  mutate(set = 'train')

## test data
test <-
  read.csv('cache/TST_LogP_3513_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(LogP, everything()) %>%
  na.omit() %>%
  mutate(set = 'test')

LogP <- rbind(train, test)

summary(LogP$LogP)

p1 <- ggplot(LogP, aes(LogP)) +
  geom_histogram(binwidth = 0.1, color = '#0B3087', fill = '#0B3087') +
  ggthemes::theme_tufte()
p1

p01 <- ggplot(LogP, aes(LogP, stat(density), colour = set)) +
  geom_freqpoly(binwidth = 0.1, size = 1) +
  scale_color_manual(values = c('#EB6B4A', '#0B3087')) +
  theme(legend.position="none")
  # ggthemes::theme_tufte()
p01

## BCF

train <-
  read.csv('cache/TR_BCF_469.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(ChemID, LogBCF) %>%
  mutate(set = 'train')

## test data
test <-
  read.csv('cache/TST_BCF_157.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(ChemID, LogBCF) %>%
  mutate(set = 'test')

BCF <- rbind(train, test)

summary(BCF$LogBCF)

p6 <- ggplot(BCF, aes(LogBCF)) +
  geom_histogram(binwidth = 0.1, color = '#0B3087', fill = '#0B3087') +
  ggthemes::theme_tufte()
p6

p06 <- ggplot(BCF, aes(LogBCF, stat(density), colour = set)) +
  geom_freqpoly(binwidth = 0.25, size = 1) +
  scale_color_manual(values = c('#EB6B4A', '#0B3087')) +
  theme(legend.position="none")
# ggthemes::theme_tufte()
p06

## LogS

## training data
train <-
  read.csv('cache/TR_WS_3158_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(LogMolar, everything()) %>%
  na.omit() %>%
  mutate(set = 'train')

## test data
test <-
  read.csv('cache/TST_WS_1066_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(LogMolar, everything()) %>%
  na.omit() %>%
  mutate(set = 'test')

LogS <- rbind(train, test)

summary(LogS$LogMolar)

p2 <- ggplot(LogS, aes(LogMolar)) +
  geom_histogram(binwidth = 0.1, color = '#0B3087', fill = '#0B3087') +
  ggthemes::theme_tufte()
p2

p02 <- ggplot(LogS, aes(LogMolar, stat(density), colour = set)) +
  geom_freqpoly(binwidth = 0.25, size = 1) +
  scale_color_manual(values = c('#EB6B4A', '#0B3087')) +
  theme(legend.position="none")
# ggthemes::theme_tufte()
p02

## Boiling Point

train <-
  read.csv('cache/TR_BP_4077_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(BP, everything()) %>%
  na.omit() %>%
  mutate(set = 'train')

## test data
test <-
  read.csv('cache/TST_BP_1358_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(BP, everything()) %>%
  na.omit() %>%
  mutate(set = 'test')

BP <- rbind(train, test)

summary(BP$BP)

p3 <- ggplot(BP, aes(BP)) +
  geom_histogram(binwidth = 5, color = '#0B3087', fill = '#0B3087') +
  ggthemes::theme_tufte()
p3

p03 <- ggplot(BP, aes(BP, stat(density), colour = set)) +
  geom_freqpoly(binwidth = 10, size = 1) +
  scale_color_manual(values = c('#EB6B4A', '#0B3087')) +
  theme(legend.position="none")
# ggthemes::theme_tufte()
p03

## Melting Point

train <-
  read.csv('cache/TR_MP_6486_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(MP, everything()) %>%
  na.omit() %>%
  mutate(set = 'train')

## test data
test <-
  read.csv('cache/TST_MP_2167_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X,-CAS,-ROMol,-SMILES,-ID) %>%
  select(MP, everything()) %>%
  na.omit() %>%
  mutate(set = 'test')

MP <- rbind(train, test)

summary(MP$MP)

p4 <- ggplot(MP, aes(MP)) +
  geom_histogram(binwidth = 5, color = '#0B3087', fill = '#0B3087') +
  ggthemes::theme_tufte()
p4

p04 <- ggplot(MP, aes(MP, stat(density), colour = set)) +
  geom_freqpoly(binwidth = 10, size = 1) +
  scale_color_manual(values = c('#EB6B4A', '#0B3087')) +
  theme(legend.position="none")
# ggthemes::theme_tufte()
p04

## Vapor Pressure

train <-
  read.csv('cache/TR_VP_2034_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X, -CAS, -ROMol, -SMILES, -ID) %>%
  select(LogVP, everything()) %>%
  na.omit() %>%
  mutate(set = 'train')

## test data
test <-
  read.csv('cache/TST_VP_679_descrs.csv',
           header = TRUE,
           stringsAsFactors = FALSE) %>%
  select(-X, -CAS, -ROMol, -SMILES, -ID) %>%
  select(LogVP, everything()) %>%
  na.omit() %>%
  mutate(set = 'test')

VP <- rbind(train, test)

summary(VP$LogVP)

p5 <- ggplot(VP, aes(LogVP)) +
  geom_histogram(binwidth = 0.1, color = '#0B3087', fill = '#0B3087') +
  ggthemes::theme_tufte()
p5

p05 <- ggplot(VP, aes(LogVP, stat(density), colour = set)) +
  geom_freqpoly(binwidth = 0.75, size = 1) +
  scale_color_manual(values = c('#EB6B4A', '#0B3087')) +
  theme(legend.position="none")
# ggthemes::theme_tufte()
p05

gridExtra::grid.arrange(p1, p2, p3, p4, p5, p6, nrow = 2)

gridExtra::grid.arrange(p01, p02, p03, p04, p05, p06, nrow = 2)