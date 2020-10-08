library(rcdk)
library(tidyr)
library(magrittr)
library(dplyr)

dc <- get.desc.categories()
dc

## hybrid descriptors
dn01 <- get.desc.names(dc[1])
dn01

## constitutional descriptors
dn02 <- get.desc.names(dc[2])
dn02

## topological descriptors
dn03 <- get.desc.names(dc[3])
dn03

## electronic descriptors
dn04 <- get.desc.names(dc[4])
dn04

## geometrical descriptors
dn05 <- get.desc.names(dc[5])
dn05

data("bpdata")
head(bpdata)

mol <- parse.smiles(bpdata$SMILES)

allDescs01 <- eval.desc(mol, dn01)
df01 <- data.frame(names(allDescs01))
df01$source <- 'dn01'
colnames(df01) <- c('descriptor', 'source')
df01$descriptor <- as.character(df01$descriptor)

allDescs02 <- eval.desc(mol, dn02)
df02 <- data.frame(names(allDescs02))
df02$source <- 'dn02'
colnames(df02) <- c('descriptor', 'source')
df02$descriptor <- as.character(df02$descriptor)

allDescs03 <- eval.desc(mol, dn03)
df03 <- data.frame(names(allDescs03))
df03$source <- 'dn03'
colnames(df03) <- c('descriptor', 'source')
df03$descriptor <- as.character(df03$descriptor)

allDescs04 <- eval.desc(mol, dn04)
df04 <- data.frame(names(allDescs04))
df04$source <- 'dn04'
colnames(df04) <- c('descriptor', 'source')
df04$descriptor <- as.character(df04$descriptor)

allDescs05 <- eval.desc(mol, dn05)
df05 <- data.frame(names(allDescs05))
df05$source <- 'dn05'
colnames(df05) <- c('descriptor', 'source')
df05$descriptor <- as.character(df05$descriptor)

allDescs <- cbind(allDescs01, allDescs02, allDescs03, allDescs04, allDescs05)

alles <- cbind(bpdata, allDescs)

descsNames <- rbind(df01, df02, df03, df04, df05)
descsNames <- data.frame(descsNames)

qaz <- descsNames %>%
  group_by(descriptor) %>%
  summarise(count = n()) %>%
  filter(count > 1)

qaz <- dplyr::full_join(df01, df02, by = 'descriptor') %>%
  dplyr::full_join(df03, by = 'descriptor') %>%
  dplyr::full_join(df04, by = 'descriptor') %>%
  dplyr::full_join(df05, by = 'descriptor')

