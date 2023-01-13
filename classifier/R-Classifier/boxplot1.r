library(tidyverse)
library(reshape)
library(ggplot2)
library(modelr)
options(na.action = na.warn)
table <- read.csv("C:/Users/z3696/Documents/Document-Classification/classifier/Output/Table.csv")
head(table)

precision =table[,5]
recall = table[,6]
classifier = table[,4]
sampling=table[,2]
technique=table[,3]
# Chapter 3 works
ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall))

ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall, color = technique))

ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall, size = technique))

# Left
ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall, alpha = technique))

# Right
ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall, shape = technique))

ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall), color = "blue")

ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall)) + facet_wrap(~ Technique, nrow = 2)

ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall)) + 
  facet_grid(Sampling ~ Technique)

ggplot(table, aes(precision, recall)) + geom_boxplot()

ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall)) +
  geom_smooth(mapping = aes(x = precision, y = recall))

ggplot(data = table) + 
  stat_summary(
    mapping = aes(x = precision, y = recall),
    fun.min = min,
    fun.max = max,
    fun = median
  )

ggplot(table, aes(x = precision, y = recall)) + 
  geom_point(size = 2, colour = "grey30") + 
  geom_abline(
    aes(intercept = a1, slope = a2, colour = -dist), 
    data = table(models, rank(dist) <= 10)
  )
ggplot(data = table, mapping = aes(x = precision, y = recall, alpha = classifier)) +
  geom_boxplot()

ggplot(data = table) +
  geom_count(mapping = aes(x = precision, y = recall, alpha = classifier))
x <- precision
y <- recall
ggplot(table) + 
  geom_histogram(mapping = aes(x = y), binwidth = 0.1)

ggplot(data = table) + 
  geom_point(mapping = aes(x = precision, y = recall))

# models <- tibble(
#   a1 = runif(250, -20, 40),
#   a2 = runif(250, -5, 5)
# )
# 7 EDA
ggplot(data = table) +
  geom_bar(mapping = aes(x = classifier))
table %>% 
  count(cut_width(Precision, 0.5))
table %>% 
  count(Classifier)
table %>% 
  count(cut_width(Recall, 0.5))
smaller <- table %>%
  filter(Classifier > 60)
ggplot(data = smaller, mapping = aes(x = Precision)) +
  geom_histogram(binwidth = 0.1)
ggplot(data = smaller, mapping = aes(x = Recall, colour = classifier)) +
  geom_freqpoly(binwidth = 0.1)
ggplot(data = smaller, mapping = aes(x = Recall)) +
  geom_histogram(binwidth = 0.01)
ggplot(data = smaller, mapping = aes(x = Precision, colour = classifier)) +
  geom_freqpoly(binwidth = 0.1)
ggplot(data = smaller, mapping = aes(x = Precision)) +
  geom_histogram(binwidth = 0.01)
ggplot(data = smaller, mapping = aes(x = Precision)) + 
  geom_histogram(binwidth = 0.25)
ggplot(table) + 
  geom_histogram(mapping = aes(x = y), binwidth = 0.5)
ggplot(table) + 
  geom_histogram(mapping = aes(x = y), binwidth = 0.5) +
  coord_cartesian(ylim = c(0, 50))
# unusual <- table %>% 
#   filter(y < 30 | y > 60) %>% 
#   select(x, y) %>%
#   arrange(y)
# unusual
ggplot(data = table, mapping = aes(x = x, y = y)) + 
  geom_point()
ggplot(data = table, mapping = aes(x = x, y = y)) + 
  geom_point(na.rm = TRUE)
ggplot(data = table, mapping = aes(x = precision)) + 
  geom_freqpoly(mapping = aes(colour = classifier), binwidth = 500)
ggplot(data = table, mapping = aes(x = precision, y = ..recall..)) + 
  geom_freqpoly(mapping = aes(colour = classifier), binwidth = 500)
ggplot(data = table, mapping = aes(x = precision, y = recall)) +
  geom_boxplot()
ggplot(data = table, mapping = aes(x = classifier, y = precision)) +
  geom_boxplot()
ggplot(data = table, mapping = aes(x = classifier, y = recall)) +
  geom_boxplot()
ggplot(data = table) +
  geom_boxplot(mapping = aes(x = reorder(classifier, precision, FUN = median), y = precision))
ggplot(data = table) +
  geom_boxplot(mapping = aes(x = reorder(classifier, precision, FUN = median), y = precision)) +
  coord_flip()
ggplot(data = table) +
  geom_count(mapping = aes(x = classifier, y = sampling))
table %>% 
  count(Classifier, Sampling)
table %>% 
  count(Classifier, Sampling) %>%  
  ggplot(mapping = aes(x = Classifier, y = Sampling)) +
  geom_tile(mapping = aes(fill = n))
ggplot(data = table) + 
  geom_point(mapping = aes(x = Precision, y = Recall), alpha = 50 / 100)
ggplot(data = smaller) +
  geom_bin2d(mapping = aes(x = Precision, y = Recall))
# install.packages("hexbin")
library(hexbin)
ggplot(data = smaller) +
  geom_hex(mapping = aes(x = Precision, y = Recall))

ggplot(data = smaller, mapping = aes(x = Precision, y = Recall)) + 
  geom_boxplot(mapping = aes(group = cut_width(Precision, 0.1)))

ggplot(data = smaller, mapping = aes(x = Precision, y = Recall)) + 
  geom_boxplot(mapping = aes(group = cut_number(Precision, 20)))

ggplot(data = table) +
  geom_point(mapping = aes(x = x, y = y)) +
  coord_cartesian(xlim = c(0, 90), ylim = c(0, 90))

ggplot(data = table) + 
  geom_point(mapping = aes(x = Precision, y = Recall))

library(modelr)

mod <- lm(log(Recall) ~ log(Precision), data = table)

table1 <- table %>% 
  add_residuals(mod) %>% 
  mutate(resid = exp(resid))

ggplot(data = table1) + 
  geom_point(mapping = aes(x = Precision, y = resid))

ggplot(data = table1) + 
  geom_boxplot(mapping = aes(x = Precision, y = resid))
ggplot(data = table, mapping = aes(x = precision)) + 
  geom_freqpoly(binwidth = 0.25)
ggplot(table, aes(Precision)) + 
  geom_freqpoly(binwidth = 0.25)
table %>% 
  count(Classifier, Sampling) %>% 
  ggplot(aes(Classifier, Sampling, fill = n)) + 
  geom_tile()
#> # A tibble: 5 x 2
#>   cut           n
#>   <ord>     <int>
#> 1 Fair       1610
#> 2 Good       4906
#> 3 Very Good 12082
#> 4 Premium   13791
#> 5 Ideal     21551
# Wrangle
table %>% 
  count(Precision) %>% 
  filter(n > 1)

table %>% 
  count(Precision, Recall) %>% 
  filter(n > 1)

table %>% 
  count(Precision, Recall) %>% 
  filter(n > 1)

# test1 <- c(precision, recall)
# test2 <- c(precision)
# str_view(test2, "60")
# Chapter 23 
x <- precision
y <- recall
# models <- tibble(
#   a1 = c(precision),
#   a2 = c(recall)
# )
models <- tibble(
  a1 = precision,
  a2 = recall
)
models
ggplot(table, aes(x=precision, y=recall)) + 
  geom_abline(aes(intercept = a1, slope = a2), data = models, alpha = 1/4) +
  geom_point()

model1 <- function(a, data) {
  a[1] + data$x * a[2]
}
model1(c(7, 1.5), table)
#>  [1]  8.5  8.5  8.5 10.0 10.0 10.0 11.5 11.5 11.5 13.0 13.0 13.0 14.5 14.5 14.5
#> [16] 16.0 16.0 16.0 17.5 17.5 17.5 19.0 19.0 19.0 20.5 20.5 20.5 22.0 22.0 22.0

measure_distance <- function(mod, data) {
  diff <- data$y - model1(mod, data)
  sqrt(mean(diff ^ 2))
}

table_dist <- function(a1, a2) {
  measure_distance(c(a1, a2), table)
}
# test2 <- c(precision[1],recall[2])
# test2
measure_distance(c(7.5, 1), table)

models <- models %>% 
  mutate(dist = purrr::map2_dbl(a1, a2, table_dist))
models
#> # A tibble: 250 x 3
#>       a1      a2  dist
#>    <dbl>   <dbl> <dbl>
#> 1 -15.2   0.0889  30.8
#> 2  30.1  -0.827   13.2
#> 3  16.0   2.27    13.2
#> 4 -10.6   1.38    18.7
#> 5 -19.6  -1.04    41.8
#> 6   7.98  4.59    19.3
#> # . with 244 more rows

ggplot(table, aes(x = precision, y = recall)) + 
  geom_point(size = 2, colour = "grey30") + 
  geom_abline(
    aes(intercept = a1, slope = a2, colour = -dist), 
    data = filter(models, rank(dist) <= 10)
  )
ggplot(models, aes(a1, a2)) +
  geom_point(data = filter(models, rank(dist) <= 10), size = 4, colour = "red") +
  geom_point(aes(colour = -dist))

grid <- expand.grid(
  a1 = seq(-5, 20, length=25, along.with=precision),
  a2 = seq(1, 3, length = 25, along.with=recall)
) %>% 
  mutate(dist = purrr::map2_dbl(a1, a2, table_dist))

grid %>% 
  ggplot(aes(a1, a2)) +
  geom_point(data = filter(grid, rank(dist) <= 10), size = 4, colour = "red") +
  geom_point(aes(colour = -dist)) 

ggplot(table, aes(x=precision, y=recall)) + 
  geom_point(size = 2, colour = "grey30") + 
  geom_abline(
    aes(intercept = a1, slope = a2, colour = -dist), 
    data = filter(grid, rank(dist) <= 10)
  )
best <- optim(c(0, 0), measure_distance, data = table)
best$par

ggplot(table, aes(x, y)) + 
  geom_point(size = 2, colour = "grey30") + 
  geom_abline(intercept = best$par[1], slope = best$par[2])

table_mod <- lm(y ~ x, data = table)
coef(table_mod)

tablea <- tibble(
  x = rep(1:10, each = 3),
  y = x * 1.5 + 6 + rt(length(x), df = 2)
)

measure_distance <- function(mod, data) {
  diff <- data$y - model1(mod, data)
  mean(abs(diff))
}

model1 <- function(a, data) {
  a[1] + data$x * a[2] + a[3]
}

grid <- table %>% 
  data_grid(x) 
grid
#> # A tibble: 10 x 1
#>       x
#>   <int>
#> 1     1
#> 2     2
#> 3     3
#> 4     4
#> 5     5
#> 6     6
#> # . with 4 more rows

grid <- grid %>% 
  add_predictions(table_mod) 
grid
#> # A tibble: 10 x 2
#>       x  pred
#>   <int> <dbl>
#> 1     1  6.27
#> 2     2  8.32
#> 3     3 10.4 
#> 4     4 12.4 
#> 5     5 14.5 
#> 6     6 16.5 
#> # . with 4 more rows

ggplot(table, aes(x)) +
  geom_point(aes(y = y)) +
  geom_line(aes(y = pred), data = grid, colour = "red", size = 1)

table <- table %>% 
  add_residuals(table_mod)
table
#> # A tibble: 30 x 3
#>       x     y  resid
#>   <int> <dbl>  <dbl>
#> 1     1  4.20 -2.07 
#> 2     1  7.51  1.24 
#> 3     1  2.13 -4.15 
#> 4     2  8.99  0.665
#> 5     2 10.2   1.92 
#> 6     2 11.3   2.97 
#> # . with 24 more rows

ggplot(table, aes(resid)) + 
  geom_freqpoly(binwidth = 0.5)

ggplot(table, aes(x, resid)) + 
  geom_ref_line(h = 0) +
  geom_point() 

ggplot(table) + 
  geom_point(aes(x, y))

mod2 <- lm(y ~ x, data = table)

grid <- table %>% 
  data_grid(x) %>% 
  add_predictions(mod2)
grid
#> # A tibble: 4 x 2
#>   x      pred
#>   <chr> <dbl>
#> 1 a      1.15
#> 2 b      8.12
#> 3 c      6.13
#> 4 d      1.91

ggplot(table, aes(x)) + 
  geom_point(aes(y = y)) +
  geom_point(data = grid, aes(y = pred), colour = "red", size = 4)

tibble(x = "e") %>% 
  add_predictions(mod2)

ggplot(table, aes(x, y)) + 
  geom_point(aes(colour = classifier))
#> Error in model.frame.default(Terms, newdata, na.action = na.action, xlev = object$xlevels): factor x has new level e

mod1 <- lm(y ~ x + classifier, data = table)
mod2 <- lm(y ~ x * classifier, data = table)

grid <- table %>% 
  data_grid(x, classifier) %>% 
  gather_predictions(mod1, mod2)
grid
# Stuck At The Moment
ggplot(table, aes(x, y, colour = classifier)) + 
  geom_point() + 
  geom_line(data = grid, aes(y = pred)) + 
  facet_wrap(~ model)


# Leftover Code
#> # A tibble: 80 x 4
#>   model    x1 x2     pred
#>   <chr> <int> <fct> <dbl>
#> 1 mod1      1 a      1.67
#> 2 mod1      1 b      4.56
#> 3 mod1      1 c      6.48
#> 4 mod1      1 d      4.03
#> 5 mod1      2 a      1.48
#> 6 mod1      2 b      4.37
#> # . with 74 more rows

# ggplot(table, aes(x, y, colour = classifier)) + 
#   geom_point() + 
#   geom_line(data = grid, aes(y = pred)) + 
#   facet_wrap(~ model)
# 
# best <- optim(c(0, 0), measure_distance, data = table)
# best$par
# #> [1] 4.222248 2.051204
# 
# data_dist <- function(x, y) {
#   measure_distance(c(x, y), table)
# }
# 
# models <- models %>% 
#   mutate(dist = purrr::map2_dbl(x, y, data_dist))
# models
# 
# 
# ggplot(table, aes(x=precision, y=recall)) + 
#   geom_point(size = 2, colour = "grey30") + 
#   geom_abline(intercept = best$par[1], slope = best$par[2])
# #ggplot(classifier, aes(color, price)) + geom_boxplot()
# #ggplot(classifier, diamonds, aes(clarity, price)) + geom_boxplot()
# 
# ggplot(table, aes(x, y)) + 
#   geom_point(size = 2, colour = "grey30") + 
#   geom_abline(
#     aes(intercept = a1, slope = a2, colour = -dist), 
#     data = filter(models, rank(dist) <= 0)
#   )
# grid <- expand.grid(
#   a1 = seq(-5, 20, length = 25),
#   a2 = seq(1, 3, length = 25)
# ) %>% 
#   mutate(dist = purrr::map2_dbl(x, y, data_dist))
# 
# grid %>% 
#   ggplot(aes(x, y)) +
#   geom_point(data = filter(grid, rank(dist) <= 10), size = 4, colour = "red") +
#   geom_point(aes(colour = -dist)) 
# 
ggplot(foo, aes(x, y, colour = classifier)) + 
  geom_point() + 
  geom_line(data = grid, aes(y = pred)) + 
  facet_wrap(~ model)

foo <- foo %>% 
  gather_residuals(mod1, mod2)

ggplot(foo, aes(x, y, colour = classifier)) + 
  geom_point() + 
  facet_grid(model ~ classifier)

mod1 <- lm(y ~ x + classifier, data = table)
mod2 <- lm(y ~ x * classifier, data = table)

grid <- foo %>% 
  data_grid(
    x = seq_range(x, 5), 
    y = seq_range(y, 5) 
  ) %>% 
  gather_predictions(mod1, mod2)
grid


ggplot(foo, aes(x, y)) + 
  geom_tile(aes(y = classifier)) + 
  facet_wrap(~ model)

ggplot(foo, aes(x, y, colour= classifier, group = classifier)) + 
  geom_line() +
  facet_wrap(~ model)
ggplot(foo, aes(y, x, colour = classifier, group = classifier)) + 
  geom_line() +
  facet_wrap(~ model)

ggplot(foo, aes(x, y)) + 
  geom_tile(aes(fill = classifier)) + 
  facet_wrap(~ model)
# Skip 23.4.4 to Chapter 24
# 24.1.1
library(tidyverse)
library(modelr)
options(na.action = na.warn)

library(nycflights13)
library(lubridate)
# 24.2
ggplot(foo, aes(x, classifier)) + geom_boxplot()
ggplot(foo, aes(y, classifier)) + geom_boxplot()
# 24.2.1
ggplot(foo, aes(x, y)) + 
  geom_hex(bins = 50)
ggplot(foo, aes(x, classifier)) + 
  geom_hex(bins = 50)
ggplot(foo, aes(y, classifier)) + 
  geom_hex(bins = 50)

foo2 <- foo2 %>% 
  add_residuals(mod_foo, "lclassifier")

ggplot(foo2, aes(x, lclassifier)) + 
  geom_hex(bins = 50)

# grid <- foo2 %>% 
#   data_grid(x = seq_range(x), 20)) %>% 
#   mutate(x = log2(x)) %>% 
#   add_predictions(mod_foo2, "l_x") %>% 
#   mutate(x = 2 ^ x)

lm1 <- lm(y ~ classifier, data=foo)
lm2 <- lm(x ~ classifier, data=foo)

ggplot(foo, aes(x, y)) + 
  geom_hex(bins = 50) + 
  geom_line(data = foo, colour = "red", size = 1)

foo2 <- foo %>%
  filter(x <= 1) %>%
  mutate(l_x = log2(l_x), l_y = log2(y))

ggplot(foo2, aes(l_x, l_y)) + 
  geom_hex(bins = 50)

mod_foo <- lm(l_x ~ l_y, data = foo2)
foo2 <- foo2 %>% 
  add_residuals(mod_foo, "l_y")

ggplot(foo2, aes(x, l_y)) + geom_boxplot()
ggplot(foo2, aes(y, l_y)) + geom_boxplot()

mod_foo2 <- lm(l_y ~ l_x, data = foo2)
grid <- foo2 %>%
  data_grid(x = seq_range(x, 225)) %>%
  mutate(l_x = log2(x)) %>%
  add_predictions(mod_foo2, "l_y") %>%
  mutate(l_y = log2(y))

foo <- foo %>% 
  add_residuals(mod_foo2, sampling)

ggplot(foo2, aes(l_x, "sampling")) + 
  geom_hex(bins = 50)

ggplot(foo2, aes(l_x, classifier)) + geom_boxplot()
ggplot(foo2, aes(l_y, classifier)) + geom_boxplot()
# 24.2.2
mod_foo2 <- lm(l_x ~ l_y + classifier + sampling + technique, data = foo)

grid <- foo2 %>% 
  data_grid(classifier, .model = mod_foo2) %>% 
  add_predictions(mod_foo2)
grid

ggplot(grid, aes(l_x, pred)) + 
  geom_point()
#> # A tibble: 5 x 5
#>   cut       lcarat color clarity  pred
#>   <ord>      <dbl> <chr> <chr>   <dbl>
#> 1 Fair      -0.515 G     VS2      11.2
#> 2 Good      -0.515 G     VS2      11.3
#> 3 Very Good -0.515 G     VS2      11.4
#> 4 Premium   -0.515 G     VS2      11.4
#> 5 Ideal     -0.515 G     VS2      11.4

ggplot(grid, aes(classifier, pred)) + 
  geom_point()

# 24.2.2

foo2 <- foo2 %>% 
  add_residuals(mod_foo, "lclassifier")

ggplot(foo2, aes(l_x, lclassifier)) + 
  geom_hex(bins = 50)

foo2 <- foo %>% 
  add_residuals(mod_foo)


ggplot(foo, aes(x, classifier)) + 
  geom_line()
# 
# foo2 %>%
# filter(abs(l_x) > 1) %>%
#  add_predictions(mod_foo) %>%
#  mutate(pred = pred) %>%
#  select(l_x, pred, l_y:all_of(foo), x:y) %>%
#  arrange(x)

# 24.3
ggplot(foo2, aes(x, classifier)) + 
  geom_line()

ggplot(foo, aes(y, classifier)) + 
  geom_line()

ggplot(foo, aes(x, y)) + 
  geom_line()
# 24.3.1
ggplot(foo, aes(classifier, x)) + 
  geom_boxplot()

ggplot(foo, aes(classifier, y)) + 
  geom_boxplot()

mod <- lm(x ~ classifier, data = foo)


grid <- foo %>% 
  data_grid(classifier) %>% 
  add_predictions(mod, "x")

ggplot(foo, aes(classifier, x)) + 
  geom_boxplot() +
  geom_point(data = grid, colour = "red", size = 4)

foo <- foo %>% 
  add_residuals(mod)

foo %>% 
  ggplot(aes(y, resid)) + 
  geom_ref_line(h = 0) + 
  geom_line()
ggplot(foo, aes(y, resid, colour = classifier)) + 
  geom_ref_line(h = 0) + 
  geom_line()

foo %>% 
  ggplot(aes(y, resid)) + 
  geom_ref_line(h = 0) + 
  geom_line(colour = "grey50") + 
  geom_smooth(se = FALSE, span = 0.20)
foo %>% 
  filter(classifier == "XGBoost") %>% 
  ggplot(aes(x, y)) + 
  geom_point() + 
  geom_line() 
foo %>% 
  filter(classifier == "DecisionTree") %>% 
  ggplot(aes(x, y)) + 
  geom_point() + 
  geom_line() 
foo %>% 
  filter(classifier == "Logistic Reg") %>% 
  ggplot(aes(x, y)) + 
  geom_point() + 
  geom_line() 
foo %>% 
  ggplot(aes(x, y, colour = classifier)) +
  geom_boxplot()

mod3 <- lm(x ~ classifier, data = foo)
mod4 <- lm(y ~ classifier, data = foo)

foo %>% 
  gather_residuals(precision = mod3, recall = mod4) %>% 
  ggplot(aes(x, resid, colour = model)) +
  geom_line(alpha = 0.75)


grid <- foo %>% 
  data_grid(x, classifier) %>% 
  add_predictions(mod2, "x")

ggplot(foo, aes(classifier, x)) +
  geom_boxplot() + 
  geom_point(data = grid, colour = "red") + 
  facet_wrap(~ classifier)

library(splines)
mod6 <- MASS::rlm(x ~ classifier, data = foo)

foo %>% 
  add_residuals(mod6, "resid") %>% 
  ggplot(aes(y, resid)) + 
  geom_hline(yintercept = 0, size = 2, colour = "white") + 
  geom_line()

foo %>% 
  data_grid(y, classifier) %>% 
  add_predictions(mod6) %>% 
  ggplot(aes(y, pred, colour = classifier)) + 
  geom_line() +
  geom_point()

# 25.1
foo %>% 
  ggplot(aes(x, y, group = classifier)) +
  geom_line(alpha = 1/3)


xg <- filter(foo2, classifier == "XGBoost")
xg %>% 
  ggplot(aes(x, y)) + 
  geom_line() + 
  ggtitle("XGBoost Full Data ")

xg_mod <- lm(x ~ y, data = foo)
xg %>% 
  add_predictions(xg_mod) %>%
  ggplot(aes(classifier, pred)) + 
  geom_line() + 
  ggtitle("XGBoost Linear trend + ")

xg %>% 
  add_residuals(xg_mod) %>% 
  ggplot(aes(x, resid)) + 
  geom_hline(yintercept = 0, colour = "white", size = 3) + 
  geom_line() + 
  ggtitle("Precision Remaining pattern")


# 25.2.2
by_classifier <- foo %>% 
  group_by(x, classifier) %>% 
  nest()

by_classifier$x[[1]]

foo_model <- function(foo) {
  lm(classifier ~ x, data = foo)
}

models <- map(by_classifier$x, foo_model)
by_classifier
foo %>% 
  add_residuals(xg_mod) %>%
  ggplot(aes(x, resid)) +
  geom_line(aes(group = classifier), alpha = 1 / 3) + 
  geom_smooth(se = FALSE)

foo %>% 
  add_residuals(xg_mod) %>%
  ggplot(aes(x, resid, group = classifier)) +
  geom_line(alpha = 1 / 3) + 
  facet_wrap(~classifier)

mod7 <- lm(y ~ x, data=foo)
glance <- broom::glance(mod7)

foo %>% 
  mutate(all_of(glance)) 
arrange(r.squared)

foo %>% 
  mutate(all_of(glance)) %>%
  ggplot(aes(classifier, r.squared)) + 
  geom_jitter(width = 0.5)


foo %>%
  mutate(all_of(glance)) %>%
  ggplot(aes(y, r.squared, colour = classifier)) +
  geom_line()



#> # A tibble: 142 x 17
#> # Groups:   country, continent [142]
#>   country continent data  model resids r.squared adj.r.squared sigma statistic
#>   <fct>   <fct>     <lis> <lis> <list>     <dbl>         <dbl> <dbl>     <dbl>
#> 1 Afghan. Asia      <tib. <lm>  <tibb.     0.948         0.942 1.22      181. 
#> 2 Albania Europe    <tib. <lm>  <tibb.     0.911         0.902 1.98      102. 
#> 3 Algeria Africa    <tib. <lm>  <tibb.     0.985         0.984 1.32      662. 
#> 4 Angola  Africa    <tib. <lm>  <tibb.     0.888         0.877 1.41       79.1
#> 5 Argent. Americas  <tib. <lm>  <tibb.     0.996         0.995 0.292    2246. 
#> 6 Austra. Oceania   <tib. <lm>  <tibb.     0.980         0.978 0.621     481. 
#> # . with 136 more rows, and 8 more variables: p.value <dbl>, df <dbl>,
#> #   logLik <dbl>, AIC <dbl>, BIC <dbl>, deviance <dbl>, df.residual <int>,
#> #   nobs <int>

foo2 %>%
  ggplot(aes(x, resid)) +
  geom_line(aes(group = country), alpha = 1 / 3) + 
  geom_smooth(se = FALSE)
#> `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'

by_country
#> # A tibble: 142 x 3
#> # Groups:   country, continent [142]
#>   country     continent data             
#>   <fct>       <fct>     <list>           
#> 1 Afghanistan Asia      <tibble [12 × 4]>
#> 2 Albania     Europe    <tibble [12 × 4]>
#> 3 Algeria     Africa    <tibble [12 × 4]>
#> 4 Angola      Africa    <tibble [12 × 4]>
#> 5 Argentina   Americas  <tibble [12 × 4]>
#> 6 Australia   Oceania   <tibble [12 × 4]>
#> # . with 136 more rows

foo2 %>% 
  ggplot(aes(x, y)) +
  geom_line(aes(group = classifier), alpha = 1 / 3) + 
  geom_smooth(se = FALSE)


#> `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
# # Alternative code
# m1 <- lm(y ~ x, data = foo2)
# grid <- data.frame(x = seq(0, 1, length = 225))
# grid %>% add_predictions(m1)
#  
# m2 <- lm(y ~ poly(x, 2), data = foo2)
# grid %>% spread_predictions(m1, m2)
# 
# grid %>% gather_predictions(m1, m2)
#  foo2 <- foo2  %>% 
#   mutate(classifier) %>% 
#   group_by() %>% 
#     summarise(n = n())
# 
# ggplot(foo, aes(x, y, group = classifier)) +
#    geom_line(alpha = 1/3)
# 
# foo <- filter(foo, classifier == "Naive Bayes")
# foo %>% 
#   ggplot(aes(x, classifier)) + 
#   geom_line() + 
#   ggtitle("Full data = ")
# 
# 
# foo2_mod <- lm(x ~ classifier, data = nz)
# foo2 %>% 
#   add_predictions(nz_mod) %>%
#   ggplot(aes(classifier, pred)) + 
#   geom_line() + 
#   ggtitle("Linear trend + ")


# Linear model 
lm (y ~ x)
lm (y ~ x + classifier + sampling + technique)
model.matrix(y ~ x)
lm(x ~ classifier)
lm(x ~ classifier - 1)
lm(formula = x ~ classifier - 1)
model.matrix(x ~ classifier)
model.matrix(x ~ classifier - 1)
lm(y ~ classifier)
lm(y ~ classifier - 1)
lm(formula = y ~ classifier - 1)
model5 <- lm(y ~ x)
summary(model5)
res <- resid(model5)
hist(res, breaks = 10, las = 1, col = 1, border = "white", xlab = "Residual",main='')
res <- resid(model5)
fit <- fitted(model5)
plot(res ~ fit, pch=19, las=1); abline(0,0, col="red")
# LM with two main effects (no interaction)
lm(y ~ x + classifier)

# LM with two main effects and interaction
lm(y ~ x * classifier)

# Same as above
lm(y ~ x + classifier + x:classifier)

# LM with interaction and no main effects
lm(y ~ x:classifier)


lmClassifier = lm(x~classifier, data = table)
summary(lmClassifier)
plot(lmClassifier, pch = 16, col = "red")
plot(lmClassifier$residuals)



lmClassifier2 = lm(y ~ classifier, data = table) #Create a linear regression with two variables
summary(lmClassifier2) #Review the results
plot(lmClassifier2, pch = 16, col = "blue")
plot(lmClassifier2$residuals)


lmClassifier3 = lm(y~classifier + sampling + technique, data = table)
summary(lmClassifier3)
plot(lmClassifier3, pch = 16, col = "green")
plot(lmClassifier3$residuals)

lmClassifier4 = lm(x~classifier + sampling + technique, data = table)
summary(lmClassifier4)
plot(lmClassifier4, pch = 16, col = "orange")
plot(lmClassifier4$residuals)

lmClassifier5 <- lm(y ~ technique, data=table)
summary(lmClassifier5)
plot(lmClassifier5, pch = 16, col = "magenta")

lmClassifier6 <- lm(x ~ technique, data=table)
summary(lmClassifier6)
plot(lmClassifier6, pch = 16, col = "purple")

