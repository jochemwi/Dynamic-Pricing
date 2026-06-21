# set wd
setwd("../data/")

# load libraries
library(lme4)
library(lmerTest)  
library(dplyr)
library(ttservice)
library(emmeans)
library(RColorBrewer)

# load data
df <- read.csv("timemeasurements_check.csv", row.names = 1)

# inspect data
head(df)
summary(df)
colnames(df)
df_sel = df[,-4]
df_sel$iterations = log10(df$iterations)
df_sel$time = log10(df$time)

par(mfrow = c(1, ncol(df_sel)),
    oma = c(4,0,0,0))
colours = brewer.pal(ncol(df_sel),'RdBu')

for (i in 1:ncol(df_sel)){
  df_name = colnames(df_sel)[i]
  
  if (df_name == 'iterations'){
    ylab = 'log10(iter)'
  }
  else{
    if (df_name == 'time'){
      ylab = 'log10(time)'
    }
    else{
      ylab = df_name
    }
  }
  
  boxplot(df_sel[,i] ~ df$method,
          data = df,
          col = colours[i],
          main = df_name,
          las = 2,
          ylab = ylab,
          xlab = NA)
}
dev.off()
par(mfrow = c(1,1))
colnames(df)

# define the formula; replicates is random effect, the others are 
# fixed effect
fit_profit <-  aov(profit ~ (M + mu + z + method)^4, data = df)
summary(fit_profit)
?aov

alias(fit_profit)
