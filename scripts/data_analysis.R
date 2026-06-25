# =============================================================================
# Author:       Chris Ambagtsheer (student number: 1216414), 
#               Jochem Widdershoven (student number: 1598228)
# Date:        24/06/2026
# Description: Create analysis figures for the model output data. 
# Usage:       Rscript data_analysis.R [output_dir]
#   Where:
#     Rscript = R program to run the script.
#     data_analysis.R = this script. 
#     output_dir = optional, output directory. Default = ./data
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0){
  output_dir = './data'
} else if (length(args) == 1) {
  output_dir = args[1]
} else {
  stop("Usage: Rscript data_analysis.R [output_dir")
}

# open the data folder for analysis
setwd(output_dir)

# load libraries
lib_path <- "./R/library"
if (!dir.exists(lib_path)) dir.create(lib_path, recursive = TRUE)
.libPaths(lib_path)

packages <- c("ggplot2", "dplyr", "patchwork", "pheatmap", "RColorBrewer", "readxl", "multcompView")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, 
                     repos   = "https://cloud.r-project.org",
                     lib     = lib_path)
    library(pkg, character.only = TRUE, lib.loc = lib_path)
  }
}

# load data
df_check <- read.csv("timemeasurements_check.csv", row.names = 1)
df_ctrl <- read.csv("timemeasurements_control.csv", row.names = 1)
df_ctrl$method = 'Ctrl'
df <- bind_rows(df_check, df_ctrl)

df$M <- paste0('M', df$M)
df$M <- factor(df$M)
df$method <- factor(df$method)
levels(df$method) <- gsub("-", "_", levels(df$method))

# assign the method labels a colour.
method_levels <- levels(df$method)
method_colours <- brewer.pal(n = length(method_levels), 'RdYlBu')
names(method_colours) = method_levels

# define the formula; replicates is random effect, the others are 
# fixed effect
fit_profit <-  aov(profit ~ M * mu * z * method, data = df)
summary(fit_profit)

#################
# post hoc test #
#################

#### PROFIT ####
jpeg(filename = 'effect_m_mu_profit.jpeg', width = 1178, height = 1208, res = 192)

par(mfrow = c(2,2), oma = c(4, 1, 2, 0))
mu_levels = sort(unique(df$mu))

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
                cex.axis = 0.5,
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
jpeg(filename = 'effect_m_mu_waste.jpeg', width = 1178, height = 1208, res = 192)

par(mfrow = c(2,2), oma = c(4, 1, 2, 0))

for (i in mu_levels){
  
  d <- droplevels(df[df$mu == i,])
  
  # Combine method and M into one grouping factor (same order boxplot() uses for method*M)
  d$grp <- interaction(d$method, d$M)
  
  # Fit ANOVA on this subset, run Tukey HSD post-hoc
  fit_i <- aov(waste ~ grp, data = d)
  tuk_i <- TukeyHSD(fit_i)
  
  # Get compact letter display
  cld_i <- multcompLetters4(fit_i, tuk_i)
  letters_df <- as.data.frame.list(cld_i$grp)
  
  # Plot boxplot using the SAME grp factor, so box order matches letters_df order
  bp <- boxplot(waste ~ grp, 
                data = d,
                las = 2,
                cex.axis = 0.5,
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
jpeg(filename = 'effect_m_mu_time.jpeg', width = 1178, height = 1208, res = 192)

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
                cex.axis = 0.5,
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
jpeg(filename = 'time_vs_mu_method.jpeg', width = 1178, height = 1208, res = 192)

df_M5 <- df[df$M == 'M5',]

# 1. Define which methods get which shapes
poly_methods   <- c("Policy iteration", "Value iteration") 
linear_methods <- setdiff(unique(df$method), poly_methods)

ggplot(df, aes(x = mu, y = time, color = method, group = method)) + 
  geom_point(size = 3) + 
  
  # Layer 1: Linear fit for the two linear methods
  stat_smooth(data = subset(df, method %in% linear_methods),
              method = "lm",
              formula = y ~ x,
              linewidth = 2,
              se = TRUE) +
  
  # Layer 2: Polynomial-like curve for the two 3-point methods
  stat_smooth(data = subset(df, method %in% poly_methods),
              method = "lm",
              formula = y ~ poly(x, 3), 
              linewidth = 2,
              se = TRUE) +
  
  scale_color_manual(values = method_colours) +
  labs(
    title = "Time vs. mu by method (M = 5)",
    x = "mu",
    y = "Time (sec)",
    color = "Method"
  )

dev.off()

### Time vs M ###
jpeg(filename = 'time_vs_m_method.jpeg', width = 1178, height = 1208, res = 192)

df_mu3 <- df[df$mu == 3,]
df_mu3$M <- as.numeric(gsub("M", "", as.character(df_mu3$M)))

ggplot(df_mu3, 
       aes(x = M, y = time, group = method, color = method)) + 
  geom_point(size = 3) +
  stat_smooth(method = "lm",
              formula = y ~ x,
              linewidth = 2,          
              se = TRUE) +
  scale_color_manual(values = method_colours) + 
  labs(title = 'M vs. time by method (mu = 3)',
       x = 'M',
       y = 'Time (sec)',
       color = 'Method') + 
  scale_x_continuous(breaks = c(5, 6))

dev.off()

### waste vs. fill_rate ### 
jpeg(filename = 'profit_vs_waste_method.jpeg', width = 1178, height = 1208, res = 192)

ggplot(df, 
       aes(x = profit, y = waste, group = method, color = method)) + 
  geom_point(size = 3) +
  stat_smooth(method = "lm",
            formula = y ~ poly(x, 3),
            linewidth = 2,           
            se = TRUE) +
  scale_color_manual(values = method_colours) + 
  labs(title = 'Profit vs. waste by method',
       x = 'Profit (cents)',
       y = 'Waste (proportion)',
       color = 'Method')

dev.off()

### fill_rate vs waste ###
jpeg(filename = 'fillrate_vs_waste_method.jpeg', width = 1178, height = 1208, res = 192)

ggplot(df, 
       aes(x = fill_rate, y = waste, group = method, color = method)) + 
  geom_point(size = 3) +
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 3),
              linewidth = 2,         
              se = TRUE) +
  scale_color_manual(values = method_colours) + 
  labs(title = 'Fill rate vs. waste by method',
       x = 'Fill rate (cents)',
       y = 'Waste (proportion)',
       color = 'Method')

dev.off()

### FDC ###
jpeg(filename = 'discount_vs_profit_fdc.jpeg', width = 1178, height = 1208, res = 192)

fdc <- read.csv("fdc_profit_waste.csv", header = TRUE)
ggplot(fdc, 
       aes(x = 0.05 * 0:10, y = profit, color = 'FDC')) + 
  geom_point(size = 3) +
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 3),
              linewidth = 2) +
  scale_color_manual(values = 'forestgreen') +
  labs(title = 'Discount vs. profit',
       x = 'Discount %',
       y = 'Profit (cent)',
       color = 'Method')

dev.off()

# discount vs waste
jpeg(filename = 'discount_vs_waste_fdc.jpeg', width = 1178, height = 1208, res = 192)

ggplot(fdc, 
       aes(x = 0.05 * 0:10, y = waste, color = 'FDC')) + 
  geom_point(size = 3) +
  stat_smooth(method = "lm",
              formula = y ~ x,
              linewidth = 2) +
  scale_color_manual(values = 'purple4') +
  labs(title = 'Discount vs. waste',
       x = 'Discount %',
       y = 'Waste (proportion)',
       color = 'Method')

dev.off()

