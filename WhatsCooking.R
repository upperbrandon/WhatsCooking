
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
  trees = 1500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

grid_of_tuning_params <- grid_regular(
  mtry(range = c(2, 4)),   
  min_n(range = c(2, 10)),
  levels = 4
)

folds <- vfold_cv(trainSet, v = 5, strata = cuisine)

rf_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

CV_results <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(roc_auc),
    control = control_grid(save_pred = FALSE, verbose = TRUE)
  )

bestTune <- select_best(CV_results, metric = "roc_auc")

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



# Boost -------------------------------------------------------------------

boost_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")


boost_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

folds <- vfold_cv(trainSet, v = 5)

grid_of_tuning_params <- grid_regular(
  trees(range = c(200, 1000)),
  tree_depth(range = c(2, 10)),
  learn_rate(range = c(0.001, 0.3)),
  levels = 4
)

CV_results <- boost_workflow %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_wf <-
  boost_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

boost_preds <- predict(final_wf, new_data=testSet, type="class") %>%
  rename(cuisine = .pred_class)

kaggle_submission_lm <- testSet %>%
  select(id) %>%
  bind_cols(boost_preds)

vroom_write(x = kaggle_submission_lm, file = "./BoostPreds.csv", delim = ",")


