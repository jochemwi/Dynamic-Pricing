# set wd
setwd("~/Wageningen University/Period 2/Advanced bioinformatics/Dynamic-Pricing/data")

# load libraries
library(lme4)
library(lmerTest)  
library(dplyr)
library(ttservice)
library(emmeans)
library(RColorBrewer)
library(dplyr)

# load data
df_check <- read.csv("timemeasurements_check.csv", row.names = 1)
df_ctrl <- read.csv("timemeasurements_control.csv", row.names = 1)
df_ctrl$method = 'Ctrl'
df <- bind_rows(df_check, df_ctrl)

df$M <- paste0('M', df$M)
df$M <- factor(df$M)
df$method <- factor(df$method)
levels(df$method) <- gsub("-", "_", levels(df$method))

# prepare data
colnames(df)
df_sel = df[,-4]
df_sel$iterations = log10(df$iterations)
df_sel$time = log10(df$time)

# assign the method labels a colour.
method_levels <- levels(df$method)
method_colours <- brewer.pal(n = length(method_levels), 'RdYlBu')
names(method_colours) = method_levels

# plot out all variables 
par(mfrow = c(1, ncol(df_sel)),
    oma = c(4,0,0,0))

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
          col = rep(method_colours[method_levels], n = ncol(df_sel)),
          main = df_name,
          las = 2,
          ylab = ylab,
          xlab = NA)
}

dev.off()

# define the formula; replicates is random effect, the others are 
# fixed effect
fit_profit <-  aov(profit ~ M * mu * z * method, data = df)
summary(fit_profit)

#################
# post hoc test #
#################

#### PROFIT ####
library(multcompView)

par(mfrow = c(2,2), oma = c(4, 1, 2, 0))
mu_levels = unique(df$mu)

for (i in mu_levels){
  
  d <- droplevels(df[df$mu == i,])
  
  # Combine method and M into one grouping factor (same order boxplot() uses for method*M)
  d$grp <- interaction(d$method, d$M)
  
  # Fit ANOVA on this subset, run Tukey HSD post-hoc
  fit_i <- aov(profit ~ grp, data = d)
  tuk_i <- TukeyHSD(fit_i)
  
  # Get compact letter display
  cld_i <- multcompLetters4(fit_i, tuk_i)
  letters_df <- as.data.frame.list(cld_i$grp)
  
  # Plot boxplot using the SAME grp factor, so box order matches letters_df order
  bp <- boxplot(profit ~ grp, 
                data = d,
                las = 2,
                cex.axis = 0.6,
                xlab = NA, 
                ylim = c(min(d$profit) - 2, max(d$profit) + 2),
                ylab = ifelse(i == 2 || i == 5, 'Profit (cents)', NA),
                col = rep(method_colours[method_levels], times = nlevels(d$M)),
                main = paste0('mu = ', i)
  )
  
  mtext(ifelse(i == 5 || i == 7, 'Method * M', NA), 
        outer = FALSE, 
        line = 7,
        side = 1,
        cex = 0.9)
  
  # Match letters to box order (boxplot's bp$names gives the order actually plotted)
  ordered_letters <- letters_df[bp$names, "Letters"]
  
  # Place letters just above each box's upper whisker
  y_offset <- diff(range(d$profit, na.rm = TRUE)) * 0.1
  text(x = 1:length(bp$names),
       y = bp$stats[5, ] + y_offset + 0.1,
       labels = ordered_letters,
       font = 2)
}

mtext('Effect of mu and M on profit', outer = TRUE, line = 0)
dev.off()

### WASTE ###
par(mfrow = c(2,2), oma = c(4, 1, 2, 0))

for (i in mu_levels){
  
  d <- droplevels(df[df$mu == i,])
  
  # Combine method and M into one grouping factor (same order boxplot() uses for method*M)
  d$grp <- interaction(d$method, d$M)
  
  # Fit ANOVA on this subset, run Tukey HSD post-hoc
  fit_i <- aov(profit ~ grp, data = d)
  tuk_i <- TukeyHSD(fit_i)
  
  # Get compact letter display
  cld_i <- multcompLetters4(fit_i, tuk_i)
  letters_df <- as.data.frame.list(cld_i$grp)
  
  # Plot boxplot using the SAME grp factor, so box order matches letters_df order
  bp <- boxplot(waste ~ grp, 
                data = d,
                las = 2,
                cex.axis = 0.6,
                xlab = NA, 
                ylim = c(min(d$waste) - 0.1 * min(d$waste), max(d$waste) + 0.1 * max(d$waste)),
                ylab = ifelse(i == 2 || i == 5, 'Waste (proportion)', NA),
                col = rep(method_colours[method_levels], times = nlevels(d$M)),
                main = paste0('mu = ', i)
  )
  
  mtext(ifelse(i == 5 || i == 7, 'Method * M', NA), 
        outer = FALSE, 
        line = 7,
        side = 1,
        cex = 0.9)
  
  # Match letters to box order (boxplot's bp$names gives the order actually plotted)
  ordered_letters <- letters_df[bp$names, "Letters"]
  
  # Place letters just above each box's upper whisker
  y_offset <- diff(range(d$waste, na.rm = TRUE)) * 0.15
  text(x = 1:length(bp$names),
       y = bp$stats[5, ] + y_offset,
       labels = ordered_letters,
       font = 2)
}

mtext('Effect of mu and M on waste', outer = TRUE, line = 0)
dev.off()

### TIME ###
par(mfrow = c(2,2), oma = c(4, 1, 2, 0))
for (i in mu_levels){
  
  d <- droplevels(df[df$mu == i,])
  
  # Combine method and M into one grouping factor (same order boxplot() uses for method*M)
  d$grp <- interaction(d$method, d$M)
  
  # Fit ANOVA on this subset, run Tukey HSD post-hoc
  fit_i <- aov(time ~ grp, data = d)
  tuk_i <- TukeyHSD(fit_i)
  
  # Get compact letter display
  cld_i <- multcompLetters4(fit_i, tuk_i)
  letters_df <- as.data.frame.list(cld_i$grp)
  
  # Plot boxplot using the SAME grp factor, so box order matches letters_df order
  bp <- boxplot(time ~ grp, 
                data = d,
                las = 2,
                cex.axis = 0.6,
                xlab = NA, 
                ylim = c(min(d$time) - 2, max(d$time) + 0.2 * max(d$time)),
                ylab = ifelse(i == 2 || i == 5, 'Time (sec)', NA),
                col = rep(method_colours[method_levels], times = nlevels(d$M)),
                main = paste0('mu = ', i)
  )
  
  mtext(ifelse(i == 5 || i == 7, 'Method * M', NA), 
        outer = FALSE, 
        line = 7,
        side = 1,
        cex = 0.9)
  
  # Match letters to box order (boxplot's bp$names gives the order actually plotted)
  ordered_letters <- letters_df[bp$names, "Letters"]
  
  # Place letters just above each box's upper whisker
  y_offset <- diff(range(d$time, na.rm = TRUE)) * 0.1
  text(x = 1:length(bp$names),
       y = bp$stats[5, ] + y_offset,
       labels = ordered_letters,
       font = 2)
}

mtext('Effect of mu and M on time', outer = TRUE, line = 0)
dev.off()

#############
# line plot #
#############

### Time vs mu ###

df_M5 <- df[df$M == 'M5',]

plot(time ~ mu,
     data = df_M5,
     xlab = 'mu',
     ylab = 'Time (min)',
     main = 'Time vs. mu by method (M = 5)',
     pch = 19,
     col = method_colours[as.character(df_M5$method)])

for (m in method_levels){
  d <- df_M5[df_M5$method == m, ]
  loess_fit <- loess(time ~ mu, data = d, span = 0.75)
  
  x_grid <- seq(min(d$mu), max(d$mu), length.out = 200)
  y_pred <- predict(loess_fit, newdata = data.frame(mu = x_grid))
  
  lines(x = x_grid, y = y_pred, col = method_colours[m], lwd = 2)
}

legend('topleft', legend = method_levels, 
       col = method_colours[method_levels], 
       pch = 19, lwd = 2)

### Time vs M ###
df_mu3 <- df[df$mu == 3,]
df_mu3$M <- as.numeric(gsub("M", "", as.character(df_mu3$M)))

plot(time ~ M,
     data = df_mu3,
     xlab = 'M',
     ylab = 'Time (min)',
     main = 'Time vs. M by method (mu = 5)',
     xaxt = 'n',
     pch = 19,
     col = method_colours[as.character(df_mu3$method)])

axis(1, at = c(5, 6))

for (m in method_levels){
  d <- df_mu3[df_mu3$method == m, ]
  lm_fit <- lm(time ~ M, data = d)
  
  x_grid <- seq(min(d$M), max(d$M), length.out = 200)
  y_pred <- predict(lm_fit, newdata = data.frame(M = x_grid))
  
  lines(x = x_grid, y = y_pred, col = method_colours[m], lwd = 2)
}

legend('topleft', legend = method_levels, 
       col = method_colours[method_levels], 
       pch = 19, lwd = 2)

dev.off()
### waste vs. fill_rate ### 
plot(waste ~ profit,
     data = df,
     xlab = 'Profit (cents)',
     ylab = 'Waste (proportion)',
     main = 'Profit vs. waste by method',
     pch = 19,
     col = method_colours[as.character(df$method)])

for (m in method_levels){
  d <- df[df$method == m, ]
  loess_fit <- loess(waste ~ profit, data = d, span = 0.99)
  
  x_grid <- seq(min(d$profit), max(d$profit), length.out = 200)
  y_pred <- predict(loess_fit, newdata = data.frame(profit = x_grid))
  
  lines(x = x_grid, y = y_pred, col = method_colours[m], lwd = 2)
}

legend('topright', legend = method_levels, 
       col = method_colours[method_levels], 
       pch = 19, lwd = 2)

### fill_rate vs waste ###
plot(waste ~ fill_rate,
     data = df,
     xlab = 'Fill rate (proportion)',
     ylab = 'Waste (proportion)',
     main = 'Fill rate vs. waste by method',
     pch = 19,
     col = method_colours[as.character(df$method)])

for (m in method_levels){
  d <- df[df$method == m, ]
  loess_fit <- loess(waste ~ fill_rate, data = d, span = 0.99)
  
  x_grid <- seq(min(d$fill_rate), max(d$fill_rate), length.out = 200)
  y_pred <- predict(loess_fit, newdata = data.frame(fill_rate = x_grid))
  
  lines(x = x_grid, y = y_pred, col = method_colours[m], lwd = 2)
}

legend('topleft', legend = method_levels, 
       col = method_colours[method_levels], 
       pch = 19, lwd = 2)
