
# Load Libraries ----------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)
library(glmnet)
library(ranger)
library(ggmosaic)
library(embed)
library(tensorflow)
library(themis) 
library(jsonlite)
library(tidytext)
library(dplyr)
library(textrecipes)
library(bonsai)
setwd("~/GitHub/WhatsCooking")

trainSet <- read_file("train.json") %>%
  fromJSON()

testSet <- read_file("test.json") %>%
  fromJSON()

# Make sure ingredients is a text column
trainSet <- trainSet %>% mutate(ingredients = map_chr(ingredients, ~ paste(.x, collapse = " ")))
testSet  <- testSet  %>% mutate(ingredients = map_chr(ingredients, ~ paste(.x, collapse = " ")))

# ---- TF-IDF Recipe ----
my_recipe <- recipe(cuisine ~ ingredients, data = trainSet) %>%
  step_tokenize(ingredients) %>%          # tokenize text
  step_tokenfilter(ingredients, max_tokens = 500) %>% 
  step_tfidf(ingredients)          


# trainSet %>%
#   unnest(ingredients) %>%
#   slice(1:10)
# 
# testSet %>%
#   unnest(ingredients) 

# Ingredient_Total <- trainSet %>%
#   mutate(ingredient_count = lengths(ingredients)) %>%
#   select(id, cuisine, ingredient_count)
# 
# Ingredient_Milk <- trainSet %>%
#   mutate(ingredient_Milk = grepl("milk", 
#                                  tolower(paste(ingredients, collapse=" ")))) %>%
#   select(id, cuisine, ingredient_Milk)
# 
# Ingredient_honey <- trainSet %>%
#   mutate(ingredient_honey = grepl("milk", 
#                                  tolower(paste(ingredients, collapse=" ")))) %>%
#   select(id, cuisine, ingredient_honey)

# 
# train_data <- trainSet %>%
#   mutate(
#     ingredient_count = lengths(ingredients),
#     ingredient_Milk = map_lgl(ingredients, ~ any(grepl("milk", tolower(.)))),
#     ingredient_Egg  = map_lgl(ingredients, ~ any(grepl("egg", tolower(.))))
#   ) %>%
#   select(id, cuisine, ingredient_count, ingredient_Milk, ingredient_Egg)
# 
# test_data <- testSet %>%
#   mutate(
#     ingredient_count = lengths(ingredients),
#     ingredient_Milk = map_lgl(ingredients, ~ any(grepl("milk", tolower(.)))),
#     ingredient_Egg  = map_lgl(ingredients, ~ any(grepl("egg", tolower(.))))
#   ) %>%
#   select(id, ingredient_count, ingredient_Milk, ingredient_Egg)
# 
# my_recipe <- recipe(cuisine ~ ., data = train_data) %>%
#   step_rm(id) %>%
#   step_dummy(all_nominal_predictors())

# RF ----------------------------------------------------------------------
my_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 400
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

grid_of_tuning_params <- grid_regular(
  mtry(range = c(2, 4)),   
  min_n(range = c(2, 10)),
  levels = 5
)

folds <- vfold_cv(trainSet, v = 5, strata = cuisine)

rf_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

CV_results <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(accuracy, roc_auc, pr_auc, f_meas, precision, recall),
    control = control_grid(save_pred = FALSE, verbose = TRUE)
  )

bestTune <- select_best(CV_results, metric = "recall")

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainSet)

cuisine_predictions <- predict(
  final_wf,
  new_data = testSet,
  type = "prob"
)

pred_class <- predict(final_wf, testSet) %>%
  rename(cuisine = .pred_class)

kaggle_submission <- testSet %>%
  select(id) %>%
  bind_cols(pred_class)

vroom_write(
  kaggle_submission,
  file = "./RFPreds.csv",
  delim = ","
)

# Light GBM

my_mod <- boost_tree(
  trees        = 100,
  tree_depth   = tune(),
  learn_rate   = tune(),
  min_n        = tune(),
  loss_reduction = tune(),
  sample_size  = tune()
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")


grid_of_tuning_params <- grid_regular(
  tree_depth(range = c(2, 10)),
  learn_rate(range = c(-3, -1)),   # log10 scale: 0.001–0.1
  min_n(range = c(2, 20)),
  loss_reduction(range = c(-5, -1)),  # gamma regularization, log10 scale
  sample_size(range = c(1, 10)),   # subsampling
  levels = 3
)


folds <- vfold_cv(trainSet, v = 3,strata = cuisine)


lgbm_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)



CV_results <- lgbm_workflow %>%
  tune_grid(
    resamples = folds,
    grid      = grid_of_tuning_params,
    metrics = metric_set(
      accuracy, roc_auc, pr_auc,
      f_meas, precision, recall
    ),
    control = control_grid(
      save_pred = FALSE,
      verbose = TRUE
    )
  )

bestTune <- select_best(CV_results, metric = "f_meas")


# ------------------------- #
# Final model
# ------------------------- #

final_wf <- lgbm_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainSet)

cuisine_predictions <- predict(
  final_wf,
  new_data = testSet,
  type = "prob"
)

pred_class <- predict(final_wf, testSet) %>%
  rename(cuisine = .pred_class)



kaggle_submission <- testSet %>%
  select(id) %>%
  bind_cols(pred_class)

vroom_write(
  kaggle_submission,
  file = "./LightGBM_Preds.csv",
  delim = ","
)


# Boost -------------------------------------------------------------------
# 
# boost_model <- boost_tree(
#   trees = tune(),
#   tree_depth = tune(),
#   learn_rate = tune()
# ) %>%
#   set_engine("lightgbm") %>%
#   set_mode("classification")
# 
# 
# boost_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(boost_model)
# 
# folds <- vfold_cv(trainSet, v = 5)
# 
# grid_of_tuning_params <- grid_regular(
#   trees(range = c(200, 1000)),
#   tree_depth(range = c(2, 10)),
#   learn_rate(range = c(0.001, 0.3)),
#   levels = 4
# )
# 
# CV_results <- boost_workflow %>%
#   tune_grid(resamples=folds,
#             grid=grid_of_tuning_params,
#             metrics=metric_set(roc_auc))
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# final_wf <-
#   boost_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainSet)
# 
# boost_preds <- predict(final_wf, new_data=testSet, type="class") %>%
#   rename(cuisine = .pred_class)
# 
# kaggle_submission_lm <- testSet %>%
#   select(id) %>%
#   bind_cols(boost_preds)
# 
# vroom_write(x = kaggle_submission_lm, file = "./BoostPreds.csv", delim = ",")
# 
# 
