Predicting the Over/Under Score for Points on NBA Players
================

Many sportsbooks across the world have options for people to place a bet
on whether a certain player will get over or under X amount of points in
a given game. We will try to find the best predictors and see if we can
accurately predict whether a player will get over or under based on
other factors.

We will examine two players, Damian Lillard and Robert Covington. Damian
Lillard is a star player for the Portland Trail Blazers and Robert
Covington is an important role player on the same team as Damian, the
Blazers.

The data was collected from basketball-reference.com and
teamrankings.com consisting of game logs dating back to the 2017-2018
season to the end of the 2020-2021 season, including postseason games if
the team made the playoffs. Game logs contain a wealth of information
such as Points, Assists, Steals and I attached more information from
teamrankings.com that contain the opponent’s defensive abilities for
each specific game log.

The data has been cleaned and transformed through a Python script and
exported into 2 csv files, Lillard_data.csv and Covington_data.csv. For
those who are not familiar with basketball stats and want to gain a
clear understanding of all the variables in these datasets, please click
here.

# Damian Lillard

Let’s start off with Damian Lillard. His Over/Under split is at 27.5
Points.

``` r
#import some packages

library(psych)
library(leaps)
library(caret)
```

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following objects are masked from 'package:psych':
    ## 
    ##     %+%, alpha

    ## Loading required package: lattice

``` r
library(tree)
library(rattle)
```

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
library(gbm)
```

    ## Loaded gbm 2.1.8

``` r
library(klaR)
```

    ## Loading required package: MASS

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## The following object is masked from 'package:psych':
    ## 
    ##     outlier

``` r
library(caTools)
library(MLeval) 
```

### Exploratory Data Analysis

``` r
#Data Cleaning

Lillard_data <- read.csv("/Users/matthew_macwan/Downloads/CIS/NBA_Over_Under_Classification/Lillard_data.csv", header = TRUE)

Lillard_data <- data.frame(Lillard_data)
Lillard_data <- na.omit(Lillard_data) 


qplot(PTS,X,data=Lillard_data)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
pairs.panels(Lillard_data[2:15])
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

``` r
pairs.panels(Lillard_data[16:29])
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-2-3.png)<!-- -->

``` r
Lillard_data$Over.Under <- as.integer(Lillard_data$Over.Under)
hist(Lillard_data$Over.Under,col="coral")
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-2-4.png)<!-- -->

``` r
prop.table(table(Lillard_data$Over.Under))
```

    ## 
    ##        0        1 
    ## 0.495114 0.504886

``` r
Lillard_data$Over.Under <- as.factor(Lillard_data$Over.Under)

levels(Lillard_data$Over.Under) <- c("Under","Over")
```

### Train/Test Split

``` r
set.seed(30) 

sample = sample.split(Lillard_data$PTS, SplitRatio = .80)
train = subset(Lillard_data, sample == TRUE)
test  = subset(Lillard_data, sample == FALSE)

#Cross Validation 

ctrl <- trainControl(method="cv", number = 5, classProbs=T, savePredictions = T)
```

Cross validation is necessary for these datasets since they are not very
large. I chose to perform K-fold cross validation with 5 folds since it
does not suffer from too much bias nor variability compared to LOOCV
approach and validation set approach.

Next, we will perform feature selection to select the variables to use
in our models.

We will try three different methods for feature selection:

-   feature importance
-   recursive feature elimination
-   best subset selection

### Feature Selection

``` r
  #rank features by importance 

control <- trainControl(method="repeatedcv", number=10, repeats=3)

feature_sel <- train(Over.Under~. - PTS - X - GmSc, data=train, method="lvq", 
               preProcess="scale", trControl=control)

importance <- varImp(feature_sel, scale=FALSE)

print(importance)
```

    ## ROC curve variable importance
    ## 
    ##   only 20 most important variables shown (out of 28)
    ## 
    ##                Importance
    ## PTS                1.0000
    ## FG                 0.9452
    ## GmSc               0.9443
    ## FGA                0.8377
    ## X3P                0.8368
    ## X3PA               0.8356
    ## FG_Percent         0.8009
    ## FTA                0.7611
    ## FT                 0.7491
    ## X3P_Percent        0.7138
    ## Opp_TO_per_POS     0.5718
    ## X                  0.5707
    ## DRB                0.5664
    ## X...               0.5619
    ## TRB                0.5495
    ## AST                0.5438
    ## Opp_Def_Eff        0.5360
    ## FT_Percent         0.5335
    ## Postseason         0.5153
    ## TOV                0.5149

``` r
plot(importance)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
#recursive feature elimination

control <- rfeControl(functions=rfFuncs, method="cv", number=10)

results <- rfe(train[c(1:20,23,24,25,26,27,28)], train[,29], sizes=c(1:15), 
               rfeControl=control)

print(results)
```

    ## 
    ## Recursive feature selection
    ## 
    ## Outer resampling method: Cross-Validated (10 fold) 
    ## 
    ## Resampling performance over subset size:
    ## 
    ##  Variables Accuracy  Kappa AccuracySD KappaSD Selected
    ##          1   0.8858 0.7713    0.05579 0.11223         
    ##          2   0.8985 0.7969    0.08716 0.17438         
    ##          3   0.9512 0.9025    0.04625 0.09220         
    ##          4   0.9555 0.9109    0.04011 0.08040         
    ##          5   0.9670 0.9340    0.03278 0.06556         
    ##          6   0.9712 0.9424    0.03942 0.07883         
    ##          7   0.9717 0.9434    0.03307 0.06605        *
    ##          8   0.9717 0.9433    0.03307 0.06623         
    ##          9   0.9675 0.9350    0.03729 0.07466         
    ##         10   0.9635 0.9270    0.04442 0.08879         
    ##         11   0.9553 0.9106    0.04815 0.09644         
    ##         12   0.9595 0.9189    0.05020 0.10053         
    ##         13   0.9595 0.9189    0.05020 0.10053         
    ##         14   0.9555 0.9110    0.05819 0.11645         
    ##         15   0.9555 0.9110    0.05819 0.11645         
    ##         26   0.9513 0.9027    0.05611 0.11228         
    ## 
    ## The top 5 variables (out of 7):
    ##    FG, FTA, FT, FG_Percent, X3PA

``` r
predictors(results)
```

    ## [1] "FG"         "FTA"        "FT"         "FG_Percent" "X3PA"      
    ## [6] "FGA"        "X3P"

``` r
plot(results, type=c("g", "o"))
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
  #best subset selection

Lillard_bss = regsubsets(Over.Under~. - PTS - X - GmSc, data=train, nvmax=7)
```

    ## Warning in leaps.setup(x, y, wt = wt, nbest = nbest, nvmax = nvmax, force.in =
    ## force.in, : 1 linear dependencies found

    ## Reordering variables and trying again:

``` r
summary(Lillard_bss)
```

    ## Subset selection object
    ## Call: regsubsets.formula(Over.Under ~ . - PTS - X - GmSc, data = train, 
    ##     nvmax = 7)
    ## 25 Variables  (and intercept)
    ##                Forced in Forced out
    ## Win_Loss           FALSE      FALSE
    ## FG                 FALSE      FALSE
    ## FGA                FALSE      FALSE
    ## FG_Percent         FALSE      FALSE
    ## X3P                FALSE      FALSE
    ## Differential       FALSE      FALSE
    ## X3PA               FALSE      FALSE
    ## X3P_Percent        FALSE      FALSE
    ## FT                 FALSE      FALSE
    ## FTA                FALSE      FALSE
    ## FT_Percent         FALSE      FALSE
    ## ORB                FALSE      FALSE
    ## DRB                FALSE      FALSE
    ## AST                FALSE      FALSE
    ## STL                FALSE      FALSE
    ## BLK                FALSE      FALSE
    ## TOV                FALSE      FALSE
    ## PF                 FALSE      FALSE
    ## X...               FALSE      FALSE
    ## Postseason         FALSE      FALSE
    ## Opp_Def_Eff        FALSE      FALSE
    ## Opp_TS             FALSE      FALSE
    ## Opp_TO_per_POS     FALSE      FALSE
    ## Opp_3PT_Made       FALSE      FALSE
    ## TRB                FALSE      FALSE
    ## 1 subsets of each size up to 8
    ## Selection Algorithm: exhaustive
    ##          Win_Loss FG  FGA FG_Percent X3P Differential X3PA X3P_Percent FT  FTA
    ## 1  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         " " " "
    ## 2  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         " " "*"
    ## 3  ( 1 ) " "      "*" " " " "        " " " "          "*"  " "         " " "*"
    ## 4  ( 1 ) " "      "*" " " " "        " " " "          "*"  "*"         " " "*"
    ## 5  ( 1 ) " "      "*" " " " "        "*" " "          "*"  "*"         " " "*"
    ## 6  ( 1 ) " "      "*" " " " "        "*" " "          "*"  "*"         " " "*"
    ## 7  ( 1 ) " "      "*" " " " "        "*" " "          "*"  "*"         "*" "*"
    ## 8  ( 1 ) " "      "*" " " " "        "*" " "          "*"  "*"         "*" "*"
    ##          FT_Percent ORB DRB TRB AST STL BLK TOV PF  X... Postseason Opp_Def_Eff
    ## 1  ( 1 ) " "        " " " " " " " " " " " " " " " " " "  " "        " "        
    ## 2  ( 1 ) " "        " " " " " " " " " " " " " " " " " "  " "        " "        
    ## 3  ( 1 ) " "        " " " " " " " " " " " " " " " " " "  " "        " "        
    ## 4  ( 1 ) " "        " " " " " " " " " " " " " " " " " "  " "        " "        
    ## 5  ( 1 ) " "        " " " " " " " " " " " " " " " " " "  " "        " "        
    ## 6  ( 1 ) " "        "*" " " " " " " " " " " " " " " " "  " "        " "        
    ## 7  ( 1 ) "*"        " " " " " " " " " " " " " " " " " "  " "        " "        
    ## 8  ( 1 ) "*"        "*" " " " " " " " " " " " " " " " "  " "        " "        
    ##          Opp_TS Opp_TO_per_POS Opp_3PT_Made
    ## 1  ( 1 ) " "    " "            " "         
    ## 2  ( 1 ) " "    " "            " "         
    ## 3  ( 1 ) " "    " "            " "         
    ## 4  ( 1 ) " "    " "            " "         
    ## 5  ( 1 ) " "    " "            " "         
    ## 6  ( 1 ) " "    " "            " "         
    ## 7  ( 1 ) " "    " "            " "         
    ## 8  ( 1 ) " "    " "            " "

#### Analysis

-   For both players, I’ve tried out each of the variables from best
    subset selection and recursive feature elimination and decided to
    use the variables from recursive feature elimination since they
    deliver a slightly higher accuracy on the test set.

-   recursive feature elimination delivers perfect accuracy on the
    logistic regression model.

### Logistic Regression

``` r
lg <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P,
            data=train, method = "glm", family="binomial",trControl=ctrl, 
            maxit = 100)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
print(lg)
```

    ## Generalized Linear Model 
    ## 
    ## 245 samples
    ##   7 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 196, 195, 197, 196, 196 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9879184  0.9758401

``` r
  #let's train the final model on the entire training set and test set 

lg_final=glm(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P, data=train, 
             family=binomial, maxit = 100)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
  #check the deviance drop off 

anova(lg_final, test="Chisq")
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Analysis of Deviance Table
    ## 
    ## Model: binomial, link: logit
    ## 
    ## Response: Over.Under
    ## 
    ## Terms added sequentially (first to last)
    ## 
    ## 
    ##            Df Deviance Resid. Df Resid. Dev  Pr(>Chi)    
    ## NULL                         244     339.61              
    ## FG          1  197.539       243     142.07 < 2.2e-16 ***
    ## FTA         1   93.644       242      48.42 < 2.2e-16 ***
    ## FT          1    3.930       241      44.49   0.04743 *  
    ## FG_Percent  1    1.244       240      43.25   0.26462    
    ## X3PA        1   43.249       239       0.00 4.821e-11 ***
    ## FGA         1    0.000       238       0.00   0.99999    
    ## X3P         1    0.000       237       0.00   0.99999    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
log.probs <- predict(lg_final, test, type="response")

predicted = ifelse(log.probs > 0.5,'Over','Under')
predicted = as.factor(predicted)
confusionMatrix(test$Over.Under,predicted)
```

    ## Warning in confusionMatrix.default(test$Over.Under, predicted): Levels are not
    ## in the same order for reference and data. Refactoring data to match.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Over Under
    ##      Over    31     0
    ##      Under    0    31
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9422, 1)
    ##     No Information Rate : 0.5        
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ##                                      
    ##             Sensitivity : 1.0        
    ##             Specificity : 1.0        
    ##          Pos Pred Value : 1.0        
    ##          Neg Pred Value : 1.0        
    ##              Prevalence : 0.5        
    ##          Detection Rate : 0.5        
    ##    Detection Prevalence : 0.5        
    ##       Balanced Accuracy : 1.0        
    ##                                      
    ##        'Positive' Class : Over       
    ## 

#### Analysis

-   There are some warning messages but they are most likely arising due
    to the small size of our datasets and some outliers in the dataset.
    Simply collecting more data would be the solution to this problem.

-   The anova test tells us how the deviance drops with each variable.
    FG, FTA, X3PA and FT are the variables that are statistically
    significant and contribute immensely in reducing deviance.

-   The Logistic Regression model predicts the Over/Under perfectly on
    the test set.

### Support Vector Machine

``` r
svc <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P,data=train, 
             method="svmLinear",preProcess = c("center","scale"),trControl=ctrl)

print(svc)
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 245 samples
    ##   7 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## Pre-processing: centered (7), scaled (7) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 196, 196, 195, 197, 196 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9919184  0.9838401
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1

``` r
  #let's perform grid search to find the optimal value for C

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 
                          1.75, 2,5))

svm_grid <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P,
                         data = train, method = "svmLinear", tuneGrid = grid,
                         preProcess = c("center", "scale"),trControl=ctrl)
```

    ## Warning: model fit failed for Fold1: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold2: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold3: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold4: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold5: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo, :
    ## There were missing values in resampled performance measures.

    ## Warning in train.default(x, y, weights = w, ...): missing values found in
    ## aggregated results

``` r
svm_grid$finalModel
```

    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: C-svc  (classification) 
    ##  parameter : cost C = 1.25 
    ## 
    ## Linear (vanilla) kernel function. 
    ## 
    ## Number of Support Vectors : 22 
    ## 
    ## Objective Function Value : -17.5713 
    ## Training error : 0.004082 
    ## Probability model included.

``` r
plot(svm_grid)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
  #let's train the final model on the entire training set and test set

final_grid <- expand.grid(C = c(1.25))

svm_final <- train(Over.Under~FG + FTA + FT + FG_Percent + X3PA + FGA + X3P, 
                  data = train, method = "svmLinear", tuneGrid = final_grid,
                  preProcess = c("center", "scale"))

svm_final
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 245 samples
    ##   7 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## Pre-processing: centered (7), scaled (7) 
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 245, 245, 245, 245, 245, 245, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9799357  0.9594251
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1.25

``` r
SVM_pred <- predict(svm_final, newdata = test)
confusionMatrix(test$Over.Under,SVM_pred)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    30    1
    ##      Over      2   29
    ##                                          
    ##                Accuracy : 0.9516         
    ##                  95% CI : (0.865, 0.9899)
    ##     No Information Rate : 0.5161         
    ##     P-Value [Acc > NIR] : 5.105e-14      
    ##                                          
    ##                   Kappa : 0.9032         
    ##                                          
    ##  Mcnemar's Test P-Value : 1              
    ##                                          
    ##             Sensitivity : 0.9375         
    ##             Specificity : 0.9667         
    ##          Pos Pred Value : 0.9677         
    ##          Neg Pred Value : 0.9355         
    ##              Prevalence : 0.5161         
    ##          Detection Rate : 0.4839         
    ##    Detection Prevalence : 0.5000         
    ##       Balanced Accuracy : 0.9521         
    ##                                          
    ##        'Positive' Class : Under          
    ## 

#### Analysis

-   After scaling the data which is recommended for support vector
    machines, the best value for C given by grid search is 1.25.

    -   C: refers to how wide or narrow we want our support vectors to
        be. As the C increases, the width between the 2 support vectors
        decreases. A smaller C is better since it generalizes better.

    -   Side note: Our dataset is not large so performing grid search is
        possible. However, as the dataset grows, a better approach is to
        use randomized search or halving grid search to reduce runtime.

-   We see that the accuracy from the training set is 0.97 and the test
    set being .951. Performing slightly worse than the logistic
    regression.

### Decision Tree

``` r
dt <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P,data=train, 
            method="rpart",trControl=ctrl)

print(dt)
```

    ## CART 
    ## 
    ## 245 samples
    ##   7 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 196, 196, 197, 196, 195 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa    
    ##   0.03719008  0.8734490  0.7464569
    ##   0.04132231  0.8817823  0.7631236
    ##   0.76859504  0.6392653  0.2699229
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.04132231.

``` r
dt_pred <- predict(dt, newdata = test)
confusionMatrix(test$Over.Under,dt_pred)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    27    4
    ##      Over      3   28
    ##                                           
    ##                Accuracy : 0.8871          
    ##                  95% CI : (0.7811, 0.9534)
    ##     No Information Rate : 0.5161          
    ##     P-Value [Acc > NIR] : 5.587e-10       
    ##                                           
    ##                   Kappa : 0.7742          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9000          
    ##             Specificity : 0.8750          
    ##          Pos Pred Value : 0.8710          
    ##          Neg Pred Value : 0.9032          
    ##              Prevalence : 0.4839          
    ##          Detection Rate : 0.4355          
    ##    Detection Prevalence : 0.5000          
    ##       Balanced Accuracy : 0.8875          
    ##                                           
    ##        'Positive' Class : Under           
    ## 

``` r
#let's plot the tree

Lillard.tree=tree(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P,train)

plot(Lillard.tree)
text(Lillard.tree, pretty=0)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
#### Analysis

-   Up to this point, we see that the decision tree performed the worst
    so far with a test set accuracy of 0.887.

-   One positive about trees compared to other machine learning
    algorithms is its ability to be explained and understood by people
    who are not data scientists. The plot of the decision tree shows
    where each node was split and at what value, ultimately leading to
    the terminal nodes of an Over or Under decision.

-   Decision trees can be improved by using boosting and bagging
    techniques which is what we will perform next.

### Stochastic Gradient Boost

``` r
sgb <- train(Over.Under~.- PTS - X - GmSc,data=train,method="gbm",trControl=ctrl)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2606             nan     0.1000    0.0553
    ##      2        1.1739             nan     0.1000    0.0439
    ##      3        1.1040             nan     0.1000    0.0369
    ##      4        1.0531             nan     0.1000    0.0231
    ##      5        0.9978             nan     0.1000    0.0271
    ##      6        0.9539             nan     0.1000    0.0222
    ##      7        0.9120             nan     0.1000    0.0179
    ##      8        0.8759             nan     0.1000    0.0155
    ##      9        0.8421             nan     0.1000    0.0114
    ##     10        0.8131             nan     0.1000    0.0127
    ##     20        0.6119             nan     0.1000    0.0061
    ##     40        0.3983             nan     0.1000    0.0013
    ##     60        0.2966             nan     0.1000    0.0015
    ##     80        0.2295             nan     0.1000    0.0007
    ##    100        0.1917             nan     0.1000   -0.0006
    ##    120        0.1571             nan     0.1000   -0.0004
    ##    140        0.1312             nan     0.1000   -0.0006
    ##    150        0.1206             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2590             nan     0.1000    0.0659
    ##      2        1.1560             nan     0.1000    0.0519
    ##      3        1.0737             nan     0.1000    0.0457
    ##      4        0.9973             nan     0.1000    0.0390
    ##      5        0.9326             nan     0.1000    0.0318
    ##      6        0.8749             nan     0.1000    0.0271
    ##      7        0.8143             nan     0.1000    0.0216
    ##      8        0.7692             nan     0.1000    0.0183
    ##      9        0.7268             nan     0.1000    0.0177
    ##     10        0.6889             nan     0.1000    0.0147
    ##     20        0.4522             nan     0.1000    0.0051
    ##     40        0.2587             nan     0.1000    0.0008
    ##     60        0.1599             nan     0.1000    0.0008
    ##     80        0.1077             nan     0.1000   -0.0001
    ##    100        0.0745             nan     0.1000   -0.0002
    ##    120        0.0556             nan     0.1000   -0.0002
    ##    140        0.0405             nan     0.1000   -0.0000
    ##    150        0.0342             nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2487             nan     0.1000    0.0721
    ##      2        1.1469             nan     0.1000    0.0441
    ##      3        1.0543             nan     0.1000    0.0377
    ##      4        0.9827             nan     0.1000    0.0284
    ##      5        0.9121             nan     0.1000    0.0298
    ##      6        0.8465             nan     0.1000    0.0302
    ##      7        0.7869             nan     0.1000    0.0263
    ##      8        0.7367             nan     0.1000    0.0199
    ##      9        0.6891             nan     0.1000    0.0193
    ##     10        0.6492             nan     0.1000    0.0170
    ##     20        0.3859             nan     0.1000    0.0058
    ##     40        0.1827             nan     0.1000    0.0022
    ##     60        0.1052             nan     0.1000   -0.0006
    ##     80        0.0645             nan     0.1000   -0.0001
    ##    100        0.0426             nan     0.1000   -0.0003
    ##    120        0.0273             nan     0.1000   -0.0001
    ##    140        0.0183             nan     0.1000    0.0000
    ##    150        0.0148             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2757             nan     0.1000    0.0590
    ##      2        1.1739             nan     0.1000    0.0485
    ##      3        1.1071             nan     0.1000    0.0322
    ##      4        1.0391             nan     0.1000    0.0340
    ##      5        0.9875             nan     0.1000    0.0214
    ##      6        0.9367             nan     0.1000    0.0250
    ##      7        0.9018             nan     0.1000    0.0137
    ##      8        0.8652             nan     0.1000    0.0198
    ##      9        0.8301             nan     0.1000    0.0170
    ##     10        0.7984             nan     0.1000    0.0142
    ##     20        0.5852             nan     0.1000    0.0034
    ##     40        0.3760             nan     0.1000    0.0031
    ##     60        0.2715             nan     0.1000    0.0004
    ##     80        0.2057             nan     0.1000   -0.0007
    ##    100        0.1644             nan     0.1000    0.0001
    ##    120        0.1330             nan     0.1000   -0.0012
    ##    140        0.1077             nan     0.1000   -0.0004
    ##    150        0.0996             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2581             nan     0.1000    0.0632
    ##      2        1.1465             nan     0.1000    0.0524
    ##      3        1.0554             nan     0.1000    0.0425
    ##      4        0.9740             nan     0.1000    0.0346
    ##      5        0.9094             nan     0.1000    0.0237
    ##      6        0.8565             nan     0.1000    0.0251
    ##      7        0.8081             nan     0.1000    0.0175
    ##      8        0.7638             nan     0.1000    0.0197
    ##      9        0.7240             nan     0.1000    0.0153
    ##     10        0.6853             nan     0.1000    0.0174
    ##     20        0.4422             nan     0.1000    0.0060
    ##     40        0.2293             nan     0.1000    0.0014
    ##     60        0.1366             nan     0.1000   -0.0001
    ##     80        0.0930             nan     0.1000   -0.0004
    ##    100        0.0664             nan     0.1000   -0.0012
    ##    120        0.0490             nan     0.1000   -0.0007
    ##    140        0.0369             nan     0.1000   -0.0001
    ##    150        0.0305             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2487             nan     0.1000    0.0626
    ##      2        1.1455             nan     0.1000    0.0459
    ##      3        1.0563             nan     0.1000    0.0414
    ##      4        0.9732             nan     0.1000    0.0424
    ##      5        0.8990             nan     0.1000    0.0328
    ##      6        0.8401             nan     0.1000    0.0262
    ##      7        0.7923             nan     0.1000    0.0183
    ##      8        0.7462             nan     0.1000    0.0191
    ##      9        0.6935             nan     0.1000    0.0239
    ##     10        0.6458             nan     0.1000    0.0197
    ##     20        0.3690             nan     0.1000    0.0055
    ##     40        0.1680             nan     0.1000    0.0028
    ##     60        0.0940             nan     0.1000   -0.0002
    ##     80        0.0556             nan     0.1000   -0.0004
    ##    100        0.0362             nan     0.1000   -0.0003
    ##    120        0.0241             nan     0.1000   -0.0002
    ##    140        0.0167             nan     0.1000   -0.0000
    ##    150        0.0136             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2738             nan     0.1000    0.0555
    ##      2        1.1821             nan     0.1000    0.0449
    ##      3        1.1182             nan     0.1000    0.0278
    ##      4        1.0794             nan     0.1000    0.0125
    ##      5        1.0143             nan     0.1000    0.0322
    ##      6        0.9624             nan     0.1000    0.0244
    ##      7        0.9246             nan     0.1000    0.0156
    ##      8        0.8819             nan     0.1000    0.0165
    ##      9        0.8491             nan     0.1000    0.0139
    ##     10        0.8196             nan     0.1000    0.0132
    ##     20        0.5972             nan     0.1000    0.0075
    ##     40        0.3911             nan     0.1000   -0.0011
    ##     60        0.2825             nan     0.1000    0.0010
    ##     80        0.2153             nan     0.1000   -0.0011
    ##    100        0.1682             nan     0.1000    0.0006
    ##    120        0.1358             nan     0.1000   -0.0001
    ##    140        0.1122             nan     0.1000   -0.0002
    ##    150        0.1032             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2656             nan     0.1000    0.0632
    ##      2        1.1630             nan     0.1000    0.0495
    ##      3        1.0655             nan     0.1000    0.0442
    ##      4        0.9927             nan     0.1000    0.0329
    ##      5        0.9245             nan     0.1000    0.0298
    ##      6        0.8605             nan     0.1000    0.0258
    ##      7        0.8095             nan     0.1000    0.0210
    ##      8        0.7634             nan     0.1000    0.0221
    ##      9        0.7275             nan     0.1000    0.0159
    ##     10        0.6904             nan     0.1000    0.0129
    ##     20        0.4387             nan     0.1000    0.0050
    ##     40        0.2260             nan     0.1000    0.0010
    ##     60        0.1423             nan     0.1000   -0.0011
    ##     80        0.0931             nan     0.1000   -0.0006
    ##    100        0.0639             nan     0.1000   -0.0001
    ##    120        0.0447             nan     0.1000    0.0001
    ##    140        0.0315             nan     0.1000   -0.0002
    ##    150        0.0270             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2493             nan     0.1000    0.0625
    ##      2        1.1394             nan     0.1000    0.0542
    ##      3        1.0467             nan     0.1000    0.0410
    ##      4        0.9712             nan     0.1000    0.0305
    ##      5        0.8953             nan     0.1000    0.0283
    ##      6        0.8438             nan     0.1000    0.0215
    ##      7        0.7923             nan     0.1000    0.0215
    ##      8        0.7435             nan     0.1000    0.0224
    ##      9        0.7032             nan     0.1000    0.0182
    ##     10        0.6680             nan     0.1000    0.0116
    ##     20        0.3966             nan     0.1000    0.0068
    ##     40        0.1742             nan     0.1000    0.0001
    ##     60        0.0960             nan     0.1000    0.0003
    ##     80        0.0548             nan     0.1000   -0.0003
    ##    100        0.0347             nan     0.1000   -0.0004
    ##    120        0.0222             nan     0.1000   -0.0005
    ##    140        0.0145             nan     0.1000   -0.0003
    ##    150        0.0119             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2645             nan     0.1000    0.0616
    ##      2        1.1670             nan     0.1000    0.0507
    ##      3        1.0806             nan     0.1000    0.0424
    ##      4        1.0101             nan     0.1000    0.0345
    ##      5        0.9739             nan     0.1000    0.0095
    ##      6        0.9178             nan     0.1000    0.0269
    ##      7        0.8646             nan     0.1000    0.0221
    ##      8        0.8301             nan     0.1000    0.0157
    ##      9        0.7921             nan     0.1000    0.0161
    ##     10        0.7615             nan     0.1000    0.0136
    ##     20        0.5557             nan     0.1000    0.0058
    ##     40        0.3612             nan     0.1000    0.0012
    ##     60        0.2692             nan     0.1000   -0.0001
    ##     80        0.2067             nan     0.1000    0.0001
    ##    100        0.1675             nan     0.1000    0.0000
    ##    120        0.1366             nan     0.1000   -0.0006
    ##    140        0.1108             nan     0.1000   -0.0005
    ##    150        0.1003             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2554             nan     0.1000    0.0637
    ##      2        1.1483             nan     0.1000    0.0515
    ##      3        1.0567             nan     0.1000    0.0418
    ##      4        0.9770             nan     0.1000    0.0368
    ##      5        0.9174             nan     0.1000    0.0281
    ##      6        0.8616             nan     0.1000    0.0282
    ##      7        0.8083             nan     0.1000    0.0245
    ##      8        0.7640             nan     0.1000    0.0181
    ##      9        0.7160             nan     0.1000    0.0194
    ##     10        0.6808             nan     0.1000    0.0153
    ##     20        0.4302             nan     0.1000    0.0030
    ##     40        0.2177             nan     0.1000    0.0003
    ##     60        0.1333             nan     0.1000   -0.0002
    ##     80        0.0899             nan     0.1000   -0.0004
    ##    100        0.0613             nan     0.1000    0.0003
    ##    120        0.0432             nan     0.1000   -0.0003
    ##    140        0.0310             nan     0.1000   -0.0003
    ##    150        0.0260             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2485             nan     0.1000    0.0637
    ##      2        1.1368             nan     0.1000    0.0555
    ##      3        1.0465             nan     0.1000    0.0411
    ##      4        0.9653             nan     0.1000    0.0360
    ##      5        0.8915             nan     0.1000    0.0335
    ##      6        0.8250             nan     0.1000    0.0256
    ##      7        0.7742             nan     0.1000    0.0220
    ##      8        0.7270             nan     0.1000    0.0239
    ##      9        0.6797             nan     0.1000    0.0174
    ##     10        0.6364             nan     0.1000    0.0149
    ##     20        0.3792             nan     0.1000    0.0048
    ##     40        0.1763             nan     0.1000    0.0007
    ##     60        0.0900             nan     0.1000   -0.0005
    ##     80        0.0530             nan     0.1000   -0.0007
    ##    100        0.0331             nan     0.1000   -0.0001
    ##    120        0.0207             nan     0.1000   -0.0003
    ##    140        0.0140             nan     0.1000   -0.0000
    ##    150        0.0118             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2755             nan     0.1000    0.0470
    ##      2        1.2003             nan     0.1000    0.0412
    ##      3        1.1358             nan     0.1000    0.0360
    ##      4        1.0785             nan     0.1000    0.0205
    ##      5        1.0214             nan     0.1000    0.0269
    ##      6        0.9678             nan     0.1000    0.0219
    ##      7        0.9304             nan     0.1000    0.0140
    ##      8        0.8935             nan     0.1000    0.0179
    ##      9        0.8596             nan     0.1000    0.0146
    ##     10        0.8259             nan     0.1000    0.0137
    ##     20        0.6136             nan     0.1000    0.0055
    ##     40        0.3983             nan     0.1000    0.0019
    ##     60        0.2958             nan     0.1000    0.0001
    ##     80        0.2332             nan     0.1000   -0.0010
    ##    100        0.1925             nan     0.1000   -0.0007
    ##    120        0.1551             nan     0.1000   -0.0007
    ##    140        0.1292             nan     0.1000   -0.0004
    ##    150        0.1176             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2650             nan     0.1000    0.0566
    ##      2        1.1627             nan     0.1000    0.0483
    ##      3        1.0681             nan     0.1000    0.0415
    ##      4        0.9916             nan     0.1000    0.0379
    ##      5        0.9216             nan     0.1000    0.0295
    ##      6        0.8704             nan     0.1000    0.0255
    ##      7        0.8156             nan     0.1000    0.0231
    ##      8        0.7700             nan     0.1000    0.0196
    ##      9        0.7234             nan     0.1000    0.0197
    ##     10        0.6852             nan     0.1000    0.0153
    ##     20        0.4431             nan     0.1000    0.0074
    ##     40        0.2429             nan     0.1000    0.0002
    ##     60        0.1524             nan     0.1000    0.0011
    ##     80        0.1106             nan     0.1000    0.0000
    ##    100        0.0765             nan     0.1000    0.0001
    ##    120        0.0548             nan     0.1000   -0.0001
    ##    140        0.0411             nan     0.1000   -0.0001
    ##    150        0.0345             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2577             nan     0.1000    0.0581
    ##      2        1.1528             nan     0.1000    0.0455
    ##      3        1.0618             nan     0.1000    0.0415
    ##      4        0.9832             nan     0.1000    0.0335
    ##      5        0.9101             nan     0.1000    0.0291
    ##      6        0.8424             nan     0.1000    0.0306
    ##      7        0.7904             nan     0.1000    0.0253
    ##      8        0.7427             nan     0.1000    0.0221
    ##      9        0.7001             nan     0.1000    0.0111
    ##     10        0.6616             nan     0.1000    0.0140
    ##     20        0.3963             nan     0.1000    0.0059
    ##     40        0.1870             nan     0.1000    0.0008
    ##     60        0.1018             nan     0.1000   -0.0002
    ##     80        0.0624             nan     0.1000    0.0007
    ##    100        0.0399             nan     0.1000    0.0002
    ##    120        0.0253             nan     0.1000   -0.0001
    ##    140        0.0173             nan     0.1000   -0.0002
    ##    150        0.0141             nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2456             nan     0.1000    0.0628
    ##      2        1.1324             nan     0.1000    0.0555
    ##      3        1.0482             nan     0.1000    0.0410
    ##      4        0.9623             nan     0.1000    0.0376
    ##      5        0.8929             nan     0.1000    0.0304
    ##      6        0.8335             nan     0.1000    0.0227
    ##      7        0.7644             nan     0.1000    0.0302
    ##      8        0.7110             nan     0.1000    0.0236
    ##      9        0.6668             nan     0.1000    0.0190
    ##     10        0.6238             nan     0.1000    0.0179
    ##     20        0.3721             nan     0.1000    0.0054
    ##     40        0.1754             nan     0.1000    0.0002
    ##     50        0.1372             nan     0.1000   -0.0007

``` r
print(sgb)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 245 samples
    ##  28 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 196, 196, 196, 196, 196 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.9183673  0.8366667
    ##   1                  100      0.9469388  0.8938808
    ##   1                  150      0.9387755  0.8775338
    ##   2                   50      0.9510204  0.9019590
    ##   2                  100      0.9387755  0.8774930
    ##   2                  150      0.9510204  0.9020407
    ##   3                   50      0.9551020  0.9101871
    ##   3                  100      0.9387755  0.8775745
    ##   3                  150      0.9510204  0.9020543
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 50, interaction.depth =
    ##  3, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
  #let's train the final model on the entire training set and test set 

final_grid <- expand.grid(n.trees = c(150), 
                          interaction.depth = c(2),
                          shrinkage = c(0.1), n.minobsinnode = 10)

sgb_final <- train(Over.Under~.- PTS - X - GmSc,data=train,method="gbm",
                   tuneGrid = final_grid)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2602             nan     0.1000    0.0604
    ##      2        1.1563             nan     0.1000    0.0479
    ##      3        1.0653             nan     0.1000    0.0422
    ##      4        0.9803             nan     0.1000    0.0363
    ##      5        0.9082             nan     0.1000    0.0336
    ##      6        0.8514             nan     0.1000    0.0243
    ##      7        0.7971             nan     0.1000    0.0231
    ##      8        0.7487             nan     0.1000    0.0243
    ##      9        0.7025             nan     0.1000    0.0208
    ##     10        0.6634             nan     0.1000    0.0178
    ##     20        0.4153             nan     0.1000    0.0074
    ##     40        0.2056             nan     0.1000    0.0019
    ##     60        0.1263             nan     0.1000   -0.0002
    ##     80        0.0839             nan     0.1000   -0.0000
    ##    100        0.0565             nan     0.1000    0.0002
    ##    120        0.0396             nan     0.1000   -0.0001
    ##    140        0.0275             nan     0.1000   -0.0001
    ##    150        0.0229             nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2443             nan     0.1000    0.0707
    ##      2        1.1261             nan     0.1000    0.0585
    ##      3        1.0287             nan     0.1000    0.0447
    ##      4        0.9411             nan     0.1000    0.0420
    ##      5        0.8704             nan     0.1000    0.0354
    ##      6        0.8107             nan     0.1000    0.0294
    ##      7        0.7488             nan     0.1000    0.0285
    ##      8        0.6992             nan     0.1000    0.0182
    ##      9        0.6568             nan     0.1000    0.0192
    ##     10        0.6160             nan     0.1000    0.0192
    ##     20        0.3651             nan     0.1000    0.0063
    ##     40        0.1760             nan     0.1000    0.0013
    ##     60        0.0998             nan     0.1000    0.0002
    ##     80        0.0625             nan     0.1000    0.0014
    ##    100        0.0405             nan     0.1000    0.0000
    ##    120        0.0274             nan     0.1000   -0.0000
    ##    140        0.0198             nan     0.1000    0.0000
    ##    150        0.0160             nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2566             nan     0.1000    0.0612
    ##      2        1.1488             nan     0.1000    0.0536
    ##      3        1.0556             nan     0.1000    0.0464
    ##      4        0.9719             nan     0.1000    0.0373
    ##      5        0.9089             nan     0.1000    0.0338
    ##      6        0.8506             nan     0.1000    0.0285
    ##      7        0.7967             nan     0.1000    0.0254
    ##      8        0.7443             nan     0.1000    0.0208
    ##      9        0.7004             nan     0.1000    0.0162
    ##     10        0.6544             nan     0.1000    0.0189
    ##     20        0.4245             nan     0.1000    0.0045
    ##     40        0.2416             nan     0.1000   -0.0004
    ##     60        0.1525             nan     0.1000   -0.0007
    ##     80        0.1027             nan     0.1000   -0.0001
    ##    100        0.0759             nan     0.1000    0.0005
    ##    120        0.0532             nan     0.1000   -0.0001
    ##    140        0.0374             nan     0.1000    0.0000
    ##    150        0.0322             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2392             nan     0.1000    0.0648
    ##      2        1.1189             nan     0.1000    0.0627
    ##      3        1.0110             nan     0.1000    0.0513
    ##      4        0.9235             nan     0.1000    0.0414
    ##      5        0.8476             nan     0.1000    0.0343
    ##      6        0.7828             nan     0.1000    0.0292
    ##      7        0.7260             nan     0.1000    0.0249
    ##      8        0.6806             nan     0.1000    0.0211
    ##      9        0.6368             nan     0.1000    0.0215
    ##     10        0.6056             nan     0.1000    0.0123
    ##     20        0.3617             nan     0.1000    0.0026
    ##     40        0.1728             nan     0.1000    0.0007
    ##     60        0.1029             nan     0.1000   -0.0002
    ##     80        0.0622             nan     0.1000    0.0001
    ##    100        0.0418             nan     0.1000    0.0001
    ##    120        0.0281             nan     0.1000    0.0001
    ##    140        0.0183             nan     0.1000   -0.0000
    ##    150        0.0153             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2210             nan     0.1000    0.0722
    ##      2        1.1075             nan     0.1000    0.0596
    ##      3        1.0231             nan     0.1000    0.0427
    ##      4        0.9514             nan     0.1000    0.0345
    ##      5        0.8918             nan     0.1000    0.0234
    ##      6        0.8417             nan     0.1000    0.0212
    ##      7        0.7881             nan     0.1000    0.0282
    ##      8        0.7455             nan     0.1000    0.0195
    ##      9        0.6962             nan     0.1000    0.0240
    ##     10        0.6534             nan     0.1000    0.0195
    ##     20        0.4187             nan     0.1000    0.0074
    ##     40        0.2058             nan     0.1000    0.0013
    ##     60        0.1246             nan     0.1000    0.0007
    ##     80        0.0798             nan     0.1000    0.0007
    ##    100        0.0565             nan     0.1000    0.0003
    ##    120        0.0386             nan     0.1000   -0.0002
    ##    140        0.0268             nan     0.1000    0.0000
    ##    150        0.0226             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2491             nan     0.1000    0.0651
    ##      2        1.1453             nan     0.1000    0.0539
    ##      3        1.0517             nan     0.1000    0.0442
    ##      4        0.9706             nan     0.1000    0.0363
    ##      5        0.9018             nan     0.1000    0.0304
    ##      6        0.8359             nan     0.1000    0.0315
    ##      7        0.7818             nan     0.1000    0.0257
    ##      8        0.7392             nan     0.1000    0.0191
    ##      9        0.6978             nan     0.1000    0.0163
    ##     10        0.6678             nan     0.1000    0.0114
    ##     20        0.4162             nan     0.1000    0.0070
    ##     40        0.2071             nan     0.1000    0.0018
    ##     60        0.1169             nan     0.1000    0.0007
    ##     80        0.0750             nan     0.1000    0.0003
    ##    100        0.0492             nan     0.1000    0.0000
    ##    120        0.0344             nan     0.1000   -0.0001
    ##    140        0.0244             nan     0.1000    0.0002
    ##    150        0.0196             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2568             nan     0.1000    0.0624
    ##      2        1.1533             nan     0.1000    0.0508
    ##      3        1.0576             nan     0.1000    0.0471
    ##      4        0.9767             nan     0.1000    0.0383
    ##      5        0.9101             nan     0.1000    0.0335
    ##      6        0.8450             nan     0.1000    0.0295
    ##      7        0.7934             nan     0.1000    0.0239
    ##      8        0.7396             nan     0.1000    0.0241
    ##      9        0.6976             nan     0.1000    0.0182
    ##     10        0.6624             nan     0.1000    0.0147
    ##     20        0.4070             nan     0.1000    0.0046
    ##     40        0.2066             nan     0.1000    0.0017
    ##     60        0.1262             nan     0.1000    0.0001
    ##     80        0.0780             nan     0.1000    0.0009
    ##    100        0.0515             nan     0.1000   -0.0000
    ##    120        0.0350             nan     0.1000   -0.0000
    ##    140        0.0238             nan     0.1000   -0.0001
    ##    150        0.0198             nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2364             nan     0.1000    0.0643
    ##      2        1.1261             nan     0.1000    0.0516
    ##      3        1.0317             nan     0.1000    0.0420
    ##      4        0.9610             nan     0.1000    0.0313
    ##      5        0.8852             nan     0.1000    0.0334
    ##      6        0.8344             nan     0.1000    0.0205
    ##      7        0.7840             nan     0.1000    0.0237
    ##      8        0.7360             nan     0.1000    0.0219
    ##      9        0.6914             nan     0.1000    0.0186
    ##     10        0.6553             nan     0.1000    0.0154
    ##     20        0.4283             nan     0.1000    0.0103
    ##     40        0.2253             nan     0.1000    0.0035
    ##     60        0.1367             nan     0.1000    0.0007
    ##     80        0.0906             nan     0.1000    0.0003
    ##    100        0.0618             nan     0.1000   -0.0004
    ##    120        0.0430             nan     0.1000   -0.0000
    ##    140        0.0299             nan     0.1000    0.0001
    ##    150        0.0260             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2542             nan     0.1000    0.0620
    ##      2        1.1488             nan     0.1000    0.0470
    ##      3        1.0488             nan     0.1000    0.0433
    ##      4        0.9786             nan     0.1000    0.0338
    ##      5        0.9057             nan     0.1000    0.0329
    ##      6        0.8453             nan     0.1000    0.0223
    ##      7        0.7937             nan     0.1000    0.0245
    ##      8        0.7441             nan     0.1000    0.0210
    ##      9        0.7011             nan     0.1000    0.0165
    ##     10        0.6682             nan     0.1000    0.0139
    ##     20        0.4247             nan     0.1000    0.0044
    ##     40        0.2203             nan     0.1000    0.0012
    ##     60        0.1323             nan     0.1000    0.0007
    ##     80        0.0855             nan     0.1000    0.0002
    ##    100        0.0565             nan     0.1000   -0.0001
    ##    120        0.0378             nan     0.1000    0.0001
    ##    140        0.0261             nan     0.1000   -0.0003
    ##    150        0.0227             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2396             nan     0.1000    0.0609
    ##      2        1.1477             nan     0.1000    0.0445
    ##      3        1.0666             nan     0.1000    0.0399
    ##      4        0.9942             nan     0.1000    0.0344
    ##      5        0.9312             nan     0.1000    0.0280
    ##      6        0.8762             nan     0.1000    0.0230
    ##      7        0.8306             nan     0.1000    0.0186
    ##      8        0.7902             nan     0.1000    0.0124
    ##      9        0.7542             nan     0.1000    0.0162
    ##     10        0.7151             nan     0.1000    0.0154
    ##     20        0.4693             nan     0.1000    0.0043
    ##     40        0.2363             nan     0.1000    0.0006
    ##     60        0.1445             nan     0.1000   -0.0000
    ##     80        0.0924             nan     0.1000    0.0003
    ##    100        0.0627             nan     0.1000    0.0000
    ##    120        0.0475             nan     0.1000    0.0002
    ##    140        0.0343             nan     0.1000    0.0000
    ##    150        0.0300             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2455             nan     0.1000    0.0642
    ##      2        1.1216             nan     0.1000    0.0568
    ##      3        1.0234             nan     0.1000    0.0473
    ##      4        0.9471             nan     0.1000    0.0368
    ##      5        0.8875             nan     0.1000    0.0271
    ##      6        0.8284             nan     0.1000    0.0282
    ##      7        0.7810             nan     0.1000    0.0199
    ##      8        0.7307             nan     0.1000    0.0229
    ##      9        0.6906             nan     0.1000    0.0169
    ##     10        0.6506             nan     0.1000    0.0182
    ##     20        0.4035             nan     0.1000    0.0035
    ##     40        0.1848             nan     0.1000    0.0016
    ##     60        0.1121             nan     0.1000   -0.0003
    ##     80        0.0724             nan     0.1000    0.0001
    ##    100        0.0477             nan     0.1000    0.0002
    ##    120        0.0310             nan     0.1000   -0.0001
    ##    140        0.0212             nan     0.1000   -0.0001
    ##    150        0.0176             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2550             nan     0.1000    0.0605
    ##      2        1.1507             nan     0.1000    0.0501
    ##      3        1.0621             nan     0.1000    0.0400
    ##      4        0.9824             nan     0.1000    0.0337
    ##      5        0.9188             nan     0.1000    0.0322
    ##      6        0.8648             nan     0.1000    0.0259
    ##      7        0.8126             nan     0.1000    0.0224
    ##      8        0.7685             nan     0.1000    0.0180
    ##      9        0.7260             nan     0.1000    0.0163
    ##     10        0.6899             nan     0.1000    0.0148
    ##     20        0.4584             nan     0.1000    0.0057
    ##     40        0.2341             nan     0.1000    0.0029
    ##     60        0.1389             nan     0.1000    0.0016
    ##     80        0.0897             nan     0.1000    0.0001
    ##    100        0.0609             nan     0.1000   -0.0003
    ##    120        0.0428             nan     0.1000    0.0003
    ##    140        0.0314             nan     0.1000   -0.0001
    ##    150        0.0280             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2359             nan     0.1000    0.0691
    ##      2        1.1292             nan     0.1000    0.0520
    ##      3        1.0368             nan     0.1000    0.0443
    ##      4        0.9533             nan     0.1000    0.0402
    ##      5        0.8801             nan     0.1000    0.0320
    ##      6        0.8196             nan     0.1000    0.0277
    ##      7        0.7642             nan     0.1000    0.0251
    ##      8        0.7174             nan     0.1000    0.0220
    ##      9        0.6780             nan     0.1000    0.0153
    ##     10        0.6382             nan     0.1000    0.0163
    ##     20        0.4054             nan     0.1000    0.0032
    ##     40        0.2093             nan     0.1000    0.0008
    ##     60        0.1314             nan     0.1000    0.0008
    ##     80        0.0855             nan     0.1000    0.0004
    ##    100        0.0609             nan     0.1000    0.0001
    ##    120        0.0420             nan     0.1000    0.0001
    ##    140        0.0323             nan     0.1000   -0.0005
    ##    150        0.0267             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2335             nan     0.1000    0.0658
    ##      2        1.1214             nan     0.1000    0.0487
    ##      3        1.0278             nan     0.1000    0.0443
    ##      4        0.9496             nan     0.1000    0.0384
    ##      5        0.8853             nan     0.1000    0.0267
    ##      6        0.8337             nan     0.1000    0.0221
    ##      7        0.7840             nan     0.1000    0.0214
    ##      8        0.7400             nan     0.1000    0.0204
    ##      9        0.6981             nan     0.1000    0.0177
    ##     10        0.6583             nan     0.1000    0.0163
    ##     20        0.4189             nan     0.1000    0.0036
    ##     40        0.2104             nan     0.1000    0.0007
    ##     60        0.1284             nan     0.1000    0.0018
    ##     80        0.0811             nan     0.1000    0.0001
    ##    100        0.0554             nan     0.1000   -0.0001
    ##    120        0.0405             nan     0.1000   -0.0001
    ##    140        0.0286             nan     0.1000   -0.0001
    ##    150        0.0241             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2503             nan     0.1000    0.0626
    ##      2        1.1476             nan     0.1000    0.0457
    ##      3        1.0569             nan     0.1000    0.0454
    ##      4        0.9790             nan     0.1000    0.0349
    ##      5        0.9121             nan     0.1000    0.0314
    ##      6        0.8583             nan     0.1000    0.0249
    ##      7        0.8122             nan     0.1000    0.0213
    ##      8        0.7554             nan     0.1000    0.0267
    ##      9        0.7087             nan     0.1000    0.0203
    ##     10        0.6642             nan     0.1000    0.0202
    ##     20        0.4177             nan     0.1000    0.0054
    ##     40        0.1967             nan     0.1000    0.0012
    ##     60        0.1172             nan     0.1000    0.0004
    ##     80        0.0738             nan     0.1000    0.0004
    ##    100        0.0486             nan     0.1000    0.0005
    ##    120        0.0339             nan     0.1000   -0.0003
    ##    140        0.0226             nan     0.1000    0.0001
    ##    150        0.0184             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2574             nan     0.1000    0.0602
    ##      2        1.1543             nan     0.1000    0.0505
    ##      3        1.0637             nan     0.1000    0.0401
    ##      4        0.9908             nan     0.1000    0.0326
    ##      5        0.9280             nan     0.1000    0.0276
    ##      6        0.8720             nan     0.1000    0.0269
    ##      7        0.8139             nan     0.1000    0.0261
    ##      8        0.7663             nan     0.1000    0.0216
    ##      9        0.7255             nan     0.1000    0.0174
    ##     10        0.6913             nan     0.1000    0.0102
    ##     20        0.4383             nan     0.1000    0.0050
    ##     40        0.2211             nan     0.1000    0.0021
    ##     60        0.1397             nan     0.1000    0.0017
    ##     80        0.0951             nan     0.1000    0.0003
    ##    100        0.0683             nan     0.1000    0.0002
    ##    120        0.0494             nan     0.1000    0.0000
    ##    140        0.0358             nan     0.1000    0.0000
    ##    150        0.0308             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2638             nan     0.1000    0.0563
    ##      2        1.1699             nan     0.1000    0.0432
    ##      3        1.0789             nan     0.1000    0.0459
    ##      4        1.0008             nan     0.1000    0.0325
    ##      5        0.9372             nan     0.1000    0.0295
    ##      6        0.8753             nan     0.1000    0.0268
    ##      7        0.8258             nan     0.1000    0.0226
    ##      8        0.7792             nan     0.1000    0.0185
    ##      9        0.7354             nan     0.1000    0.0169
    ##     10        0.6893             nan     0.1000    0.0215
    ##     20        0.4327             nan     0.1000    0.0083
    ##     40        0.2168             nan     0.1000    0.0018
    ##     60        0.1219             nan     0.1000    0.0013
    ##     80        0.0757             nan     0.1000    0.0002
    ##    100        0.0475             nan     0.1000    0.0000
    ##    120        0.0321             nan     0.1000    0.0000
    ##    140        0.0227             nan     0.1000    0.0000
    ##    150        0.0188             nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2430             nan     0.1000    0.0661
    ##      2        1.1380             nan     0.1000    0.0547
    ##      3        1.0448             nan     0.1000    0.0455
    ##      4        0.9665             nan     0.1000    0.0401
    ##      5        0.8921             nan     0.1000    0.0353
    ##      6        0.8389             nan     0.1000    0.0236
    ##      7        0.7708             nan     0.1000    0.0326
    ##      8        0.7211             nan     0.1000    0.0196
    ##      9        0.6856             nan     0.1000    0.0161
    ##     10        0.6502             nan     0.1000    0.0163
    ##     20        0.3872             nan     0.1000    0.0074
    ##     40        0.1845             nan     0.1000    0.0007
    ##     60        0.1096             nan     0.1000    0.0004
    ##     80        0.0684             nan     0.1000   -0.0001
    ##    100        0.0441             nan     0.1000    0.0002
    ##    120        0.0305             nan     0.1000    0.0001
    ##    140        0.0208             nan     0.1000    0.0000
    ##    150        0.0173             nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2480             nan     0.1000    0.0662
    ##      2        1.1351             nan     0.1000    0.0547
    ##      3        1.0460             nan     0.1000    0.0457
    ##      4        0.9636             nan     0.1000    0.0371
    ##      5        0.8987             nan     0.1000    0.0312
    ##      6        0.8405             nan     0.1000    0.0281
    ##      7        0.7931             nan     0.1000    0.0208
    ##      8        0.7547             nan     0.1000    0.0150
    ##      9        0.7147             nan     0.1000    0.0168
    ##     10        0.6737             nan     0.1000    0.0153
    ##     20        0.4114             nan     0.1000    0.0046
    ##     40        0.2014             nan     0.1000    0.0009
    ##     60        0.1226             nan     0.1000    0.0003
    ##     80        0.0780             nan     0.1000   -0.0001
    ##    100        0.0507             nan     0.1000   -0.0001
    ##    120        0.0341             nan     0.1000   -0.0002
    ##    140        0.0245             nan     0.1000   -0.0000
    ##    150        0.0209             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2594             nan     0.1000    0.0617
    ##      2        1.1510             nan     0.1000    0.0509
    ##      3        1.0495             nan     0.1000    0.0462
    ##      4        0.9720             nan     0.1000    0.0344
    ##      5        0.9006             nan     0.1000    0.0323
    ##      6        0.8417             nan     0.1000    0.0262
    ##      7        0.7889             nan     0.1000    0.0252
    ##      8        0.7458             nan     0.1000    0.0201
    ##      9        0.7039             nan     0.1000    0.0181
    ##     10        0.6658             nan     0.1000    0.0162
    ##     20        0.4194             nan     0.1000    0.0066
    ##     40        0.2155             nan     0.1000    0.0008
    ##     60        0.1336             nan     0.1000   -0.0007
    ##     80        0.0841             nan     0.1000    0.0001
    ##    100        0.0575             nan     0.1000    0.0001
    ##    120        0.0378             nan     0.1000    0.0001
    ##    140        0.0257             nan     0.1000   -0.0000
    ##    150        0.0213             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2336             nan     0.1000    0.0629
    ##      2        1.1214             nan     0.1000    0.0521
    ##      3        1.0250             nan     0.1000    0.0448
    ##      4        0.9495             nan     0.1000    0.0374
    ##      5        0.8819             nan     0.1000    0.0324
    ##      6        0.8103             nan     0.1000    0.0346
    ##      7        0.7634             nan     0.1000    0.0238
    ##      8        0.7208             nan     0.1000    0.0203
    ##      9        0.6717             nan     0.1000    0.0233
    ##     10        0.6361             nan     0.1000    0.0161
    ##     20        0.3994             nan     0.1000    0.0072
    ##     40        0.2021             nan     0.1000    0.0008
    ##     60        0.1212             nan     0.1000   -0.0004
    ##     80        0.0781             nan     0.1000   -0.0004
    ##    100        0.0529             nan     0.1000   -0.0003
    ##    120        0.0369             nan     0.1000    0.0001
    ##    140        0.0274             nan     0.1000    0.0000
    ##    150        0.0226             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2765             nan     0.1000    0.0518
    ##      2        1.1789             nan     0.1000    0.0459
    ##      3        1.0843             nan     0.1000    0.0440
    ##      4        1.0154             nan     0.1000    0.0324
    ##      5        0.9428             nan     0.1000    0.0339
    ##      6        0.8832             nan     0.1000    0.0267
    ##      7        0.8381             nan     0.1000    0.0178
    ##      8        0.7901             nan     0.1000    0.0208
    ##      9        0.7451             nan     0.1000    0.0191
    ##     10        0.7082             nan     0.1000    0.0152
    ##     20        0.4318             nan     0.1000    0.0086
    ##     40        0.2239             nan     0.1000    0.0016
    ##     60        0.1352             nan     0.1000   -0.0003
    ##     80        0.0864             nan     0.1000    0.0011
    ##    100        0.0602             nan     0.1000    0.0002
    ##    120        0.0408             nan     0.1000   -0.0001
    ##    140        0.0284             nan     0.1000   -0.0001
    ##    150        0.0237             nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2444             nan     0.1000    0.0692
    ##      2        1.1312             nan     0.1000    0.0536
    ##      3        1.0417             nan     0.1000    0.0414
    ##      4        0.9574             nan     0.1000    0.0419
    ##      5        0.8823             nan     0.1000    0.0344
    ##      6        0.8203             nan     0.1000    0.0291
    ##      7        0.7621             nan     0.1000    0.0254
    ##      8        0.7129             nan     0.1000    0.0252
    ##      9        0.6696             nan     0.1000    0.0167
    ##     10        0.6291             nan     0.1000    0.0176
    ##     20        0.3635             nan     0.1000    0.0079
    ##     40        0.1617             nan     0.1000    0.0010
    ##     60        0.0862             nan     0.1000   -0.0001
    ##     80        0.0558             nan     0.1000    0.0000
    ##    100        0.0366             nan     0.1000   -0.0000
    ##    120        0.0247             nan     0.1000   -0.0001
    ##    140        0.0161             nan     0.1000   -0.0000
    ##    150        0.0129             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2436             nan     0.1000    0.0653
    ##      2        1.1300             nan     0.1000    0.0549
    ##      3        1.0305             nan     0.1000    0.0457
    ##      4        0.9492             nan     0.1000    0.0410
    ##      5        0.8804             nan     0.1000    0.0328
    ##      6        0.8223             nan     0.1000    0.0259
    ##      7        0.7686             nan     0.1000    0.0266
    ##      8        0.7262             nan     0.1000    0.0154
    ##      9        0.6847             nan     0.1000    0.0165
    ##     10        0.6439             nan     0.1000    0.0175
    ##     20        0.4000             nan     0.1000    0.0056
    ##     40        0.1985             nan     0.1000    0.0034
    ##     60        0.1193             nan     0.1000    0.0006
    ##     80        0.0780             nan     0.1000   -0.0002
    ##    100        0.0514             nan     0.1000   -0.0002
    ##    120        0.0368             nan     0.1000   -0.0000
    ##    140        0.0266             nan     0.1000    0.0002
    ##    150        0.0224             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2624             nan     0.1000    0.0644
    ##      2        1.1647             nan     0.1000    0.0449
    ##      3        1.0785             nan     0.1000    0.0415
    ##      4        0.9994             nan     0.1000    0.0373
    ##      5        0.9342             nan     0.1000    0.0301
    ##      6        0.8781             nan     0.1000    0.0255
    ##      7        0.8306             nan     0.1000    0.0194
    ##      8        0.7833             nan     0.1000    0.0212
    ##      9        0.7429             nan     0.1000    0.0170
    ##     10        0.6930             nan     0.1000    0.0206
    ##     20        0.4231             nan     0.1000    0.0110
    ##     40        0.2143             nan     0.1000    0.0022
    ##     60        0.1249             nan     0.1000    0.0001
    ##     80        0.0785             nan     0.1000   -0.0001
    ##    100        0.0538             nan     0.1000    0.0001
    ##    120        0.0374             nan     0.1000   -0.0001
    ##    140        0.0259             nan     0.1000    0.0001
    ##    150        0.0220             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2614             nan     0.1000    0.0598
    ##      2        1.1570             nan     0.1000    0.0491
    ##      3        1.0709             nan     0.1000    0.0417
    ##      4        0.9981             nan     0.1000    0.0367
    ##      5        0.9296             nan     0.1000    0.0302
    ##      6        0.8688             nan     0.1000    0.0282
    ##      7        0.8293             nan     0.1000    0.0155
    ##      8        0.7842             nan     0.1000    0.0189
    ##      9        0.7441             nan     0.1000    0.0200
    ##     10        0.7034             nan     0.1000    0.0194
    ##     20        0.4489             nan     0.1000    0.0046
    ##     40        0.2384             nan     0.1000    0.0006
    ##     60        0.1463             nan     0.1000    0.0006
    ##     80        0.1053             nan     0.1000    0.0010
    ##    100        0.0747             nan     0.1000   -0.0001
    ##    120        0.0555             nan     0.1000   -0.0002
    ##    140        0.0426             nan     0.1000   -0.0002
    ##    150        0.0363             nan     0.1000   -0.0001

``` r
summary(sgb_final)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

    ##                           var     rel.inf
    ## FG                         FG 59.47558592
    ## FT                         FT 11.27262458
    ## FTA                       FTA  9.85529863
    ## X3PA                     X3PA  7.13532406
    ## X3P                       X3P  4.97975725
    ## FG_Percent         FG_Percent  1.65089453
    ## FGA                       FGA  1.42052442
    ## DRB                       DRB  1.34317902
    ## Opp_Def_Eff       Opp_Def_Eff  0.67462437
    ## X3P_Percent       X3P_Percent  0.42981589
    ## TRB                       TRB  0.36020247
    ## TOV                       TOV  0.27289446
    ## BLK                       BLK  0.24502798
    ## X...                     X...  0.21877214
    ## Differential     Differential  0.14509502
    ## FT_Percent         FT_Percent  0.12055317
    ## Opp_3PT_Made     Opp_3PT_Made  0.10728462
    ## Opp_TS                 Opp_TS  0.10017880
    ## AST                       AST  0.08502184
    ## Postseason         Postseason  0.04474236
    ## ORB                       ORB  0.02373993
    ## STL                       STL  0.01692649
    ## PF                         PF  0.01128871
    ## Opp_TO_per_POS Opp_TO_per_POS  0.01064333
    ## Win_Loss             Win_Loss  0.00000000

``` r
sgb.predict <- predict(sgb_final, newdata = test)
confusionMatrix(test$Over.Under,sgb.predict)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    30    1
    ##      Over      1   30
    ##                                           
    ##                Accuracy : 0.9677          
    ##                  95% CI : (0.8883, 0.9961)
    ##     No Information Rate : 0.5             
    ##     P-Value [Acc > NIR] : 4.237e-16       
    ##                                           
    ##                   Kappa : 0.9355          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9677          
    ##             Specificity : 0.9677          
    ##          Pos Pred Value : 0.9677          
    ##          Neg Pred Value : 0.9677          
    ##              Prevalence : 0.5000          
    ##          Detection Rate : 0.4839          
    ##    Detection Prevalence : 0.5000          
    ##       Balanced Accuracy : 0.9677          
    ##                                           
    ##        'Positive' Class : Under           
    ## 

#### Analysis

-   Grid search gives us the optimal hyper-parameters for the final
    model which is n.trees = 150, interaction.depth = 3, shrinkage =
    0.1. The accuracy was 0.9677.

    -   n.trees: The number of trees built.

    -   interaction.depth: the depth of each tree.

    -   shrinkage: the learning rate. A smaller shrinkage value is
        always better than a larger one but this comes at a trade off of
        runtime and storage. Choose an ideal value for shrinkage. 0.1
        works well in this situation.

-   The test set accuracy was slightly better at 0.9677.

### Random Forest

``` r
rf <- train(Over.Under~.- PTS - X - GmSc,data=train,method="rf",trControl=ctrl)

print(rf)
```

    ## Random Forest 
    ## 
    ## 245 samples
    ##  28 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 196, 196, 197, 195, 196 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9390986  0.8782139
    ##   13    0.9351837  0.8703741
    ##   25    0.9231837  0.8463060
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

``` r
  #let's perform grid search to find more optimal values for 'mtry'

tunegrid <- expand.grid(.mtry = (1:15)) 

rf_grid <- train(Over.Under ~.- PTS - X, data = train, method = 'rf',
                       tuneGrid = tunegrid,trControl=ctrl)

rf_grid$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x))) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 11
    ## 
    ##         OOB estimate of  error rate: 4.49%
    ## Confusion matrix:
    ##       Under Over class.error
    ## Under   114    7  0.05785124
    ## Over      4  120  0.03225806

``` r
  #let's train the final model on the entire training set and test set 

rf_final <- randomForest(Over.Under ~ .- PTS - X - GmSc, data = train, mtry = 2)

rf.predict <- predict(rf_final, test, type = "response")
confusionMatrix(test$Over.Under,rf.predict)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    30    1
    ##      Over      1   30
    ##                                           
    ##                Accuracy : 0.9677          
    ##                  95% CI : (0.8883, 0.9961)
    ##     No Information Rate : 0.5             
    ##     P-Value [Acc > NIR] : 4.237e-16       
    ##                                           
    ##                   Kappa : 0.9355          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9677          
    ##             Specificity : 0.9677          
    ##          Pos Pred Value : 0.9677          
    ##          Neg Pred Value : 0.9677          
    ##              Prevalence : 0.5000          
    ##          Detection Rate : 0.4839          
    ##    Detection Prevalence : 0.5000          
    ##       Balanced Accuracy : 0.9677          
    ##                                           
    ##        'Positive' Class : Under           
    ## 

``` r
#let's plot ROC curves of the Decision Tree, SGB and Random Forest to evaluate 
#how the bagging and boosting techniques improved model performance

roc_plots <- evalm(list(dt,sgb,rf_grid),gnames=c('Decison Tree','Stochastic GB',
                                                 'Random Forest'))
```

    ## ***MLeval: Machine Learning Model Evaluation***

    ## Input: caret train function object

    ## Not averaging probs.

    ## Group 1 type: cv

    ## Group 2 type: cv

    ## Group 3 type: cv

    ## Observations: 735

    ## Number of groups: 3

    ## Observations per group: 245

    ## Positive: Over

    ## Negative: Under

    ## Group: Decison Tree

    ## Positive: 124

    ## Negative: 121

    ## Group: Stochastic GB

    ## Positive: 124

    ## Negative: 121

    ## Group: Random Forest

    ## Positive: 124

    ## Negative: 121

    ## ***Performance Metrics***

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-17-2.png)<!-- -->![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-17-3.png)<!-- -->

    ## Decison Tree Optimal Informedness = 0.770994401492935

    ## Stochastic GB Optimal Informedness = 0.918555051986137

    ## Random Forest Optimal Informedness = 0.926219674753399

    ## Decison Tree AUC-ROC = 0.87

    ## Stochastic GB AUC-ROC = 0.99

    ## Random Forest AUC-ROC = 0.99

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-17-4.png)<!-- -->
#### Analysis

-   Grid search found the optimal value for mtry = 2. We see that
    accuracy slightly decreases as mtry increases.

    -   mtry: number of variables randomly sampled at each split.

-   The test set for the random forest model is 0.9677.

-   We also plotted the some curves for the decision tree, stochastic
    gradient boost model and the random forest algorithm. Let’s focus in
    on the following 2:

    -   ROC Curves: We see that the SGB and Random forest models offer a
        much better ratio at around 90% recall and 5% false positive
        rate. The decision tree clearly performs worse than the other 2.

    -   Precision/Recall Curves: We see that the random forest model has
        the highest area under the curve (AUC) at 0.95. Precision starts
        to fall rapidly at about 80% recall so we will probably select
        the precision/recall tradeoff just before that drop.

        -   Side Note: Depending on the situation or your particular
            need, you may want a threshold that delivers higher
            precision or recall at the expense of the other, for our
            need, we will take a model with the most equal tradeoff.

# Robert Covington

Next, we have Robert Covington. His Over/Under split is at 11.5 Points.

### Exploratory Data Analysis

``` r
Covington_data <- read.csv("/Users/matthew_macwan/Downloads/CIS/NBA_Over_Under_Classification/Covington_data.csv", header = TRUE)

Covington_data <- data.frame(Covington_data)
Covington_data <- na.omit(Covington_data) 

qplot(PTS,X,data=Covington_data)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
pairs.panels(Covington_data[2:15])
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-18-2.png)<!-- -->

``` r
pairs.panels(Covington_data[16:29])
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-18-3.png)<!-- -->

``` r
Covington_data$Over.Under <- as.integer(Covington_data$Over.Under)
hist(Covington_data$Over.Under,col="coral")
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-18-4.png)<!-- -->

``` r
prop.table(table(Covington_data$Over.Under))
```

    ## 
    ##         0         1 
    ## 0.4487179 0.5512821

``` r
Covington_data$Over.Under <- as.factor(Covington_data$Over.Under)

  #let's rename the factor levels 

levels(Covington_data$Over.Under) <- c("Under","Over")
```

### Train/Test Split

``` r
set.seed(35) 

sample = sample.split(Covington_data$PTS, SplitRatio = .80)
train_2 = subset(Covington_data, sample == TRUE)
test_2  = subset(Covington_data, sample == FALSE)

#Cross Validation Set

ctrl <- trainControl(method="cv", number = 5, classProbs=T, savePredictions = T)
```

### Feature Selection

``` r
#rank features by importance 

control <- trainControl(method="repeatedcv", number=10, repeats=3)

feature_sel <- train(Over.Under~., data=train_2, method="lvq", 
                     preProcess="scale", trControl=control)

importance <- varImp(feature_sel, scale=FALSE)

print(importance)
```

    ## ROC curve variable importance
    ## 
    ##   only 20 most important variables shown (out of 28)
    ## 
    ##              Importance
    ## PTS              1.0000
    ## FG               0.9760
    ## GmSc             0.9381
    ## X3P              0.9071
    ## FGA              0.8702
    ## X3PA             0.8230
    ## X3P_Percent      0.7702
    ## FG_Percent       0.7643
    ## STL              0.6430
    ## FTA              0.5946
    ## PF               0.5765
    ## FT               0.5712
    ## Plus_Minus       0.5695
    ## BLK              0.5626
    ## TRB              0.5619
    ## X                0.5597
    ## DRB              0.5597
    ## ORB              0.5514
    ## Opp_3PT_Made     0.5365
    ## Postseason       0.5339

``` r
plot(importance)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
#recursive feature elimination

control <- rfeControl(functions=rfFuncs, method="cv", number=10)

results <- rfe(train_2[c(1:20,23,24,25,26,27,28)], train_2[,29], sizes=c(1:15), 
               rfeControl=control)

print(results)
```

    ## 
    ## Recursive feature selection
    ## 
    ## Outer resampling method: Cross-Validated (10 fold) 
    ## 
    ## Resampling performance over subset size:
    ## 
    ##  Variables Accuracy  Kappa AccuracySD KappaSD Selected
    ##          1   0.9282 0.8511    0.08119  0.1708         
    ##          2   0.9128 0.8209    0.08064  0.1683         
    ##          3   0.9359 0.8704    0.05190  0.1029         
    ##          4   0.9353 0.8673    0.06526  0.1349         
    ##          5   0.9199 0.8356    0.07568  0.1556         
    ##          6   0.9199 0.8352    0.06643  0.1375         
    ##          7   0.9519 0.9033    0.05712  0.1132         
    ##          8   0.9442 0.8880    0.06593  0.1311         
    ##          9   0.9442 0.8876    0.05506  0.1091         
    ##         10   0.9365 0.8723    0.06313  0.1255         
    ##         11   0.9365 0.8723    0.06313  0.1255         
    ##         12   0.9526 0.9034    0.04090  0.0834        *
    ##         13   0.9365 0.8705    0.06313  0.1288         
    ##         14   0.9365 0.8705    0.06313  0.1288         
    ##         15   0.9288 0.8545    0.06933  0.1419         
    ##         26   0.9288 0.8563    0.06933  0.1391         
    ## 
    ## The top 5 variables (out of 12):
    ##    FG, FGA, X3P, FG_Percent, X3P_Percent

``` r
predictors(results)
```

    ##  [1] "FG"           "FGA"          "X3P"          "FG_Percent"   "X3P_Percent" 
    ##  [6] "X3PA"         "FT"           "FTA"          "Opp_3PT_Made" "BLK"         
    ## [11] "STL"          "X"

``` r
plot(results, type=c("g", "o"))
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

``` r
#best subset selection

Cov_bss = regsubsets(Over.Under~. - PTS - X - GmSc, data=train_2, nvmax=7)
```

    ## Warning in leaps.setup(x, y, wt = wt, nbest = nbest, nvmax = nvmax, force.in =
    ## force.in, : 1 linear dependencies found

    ## Reordering variables and trying again:

``` r
summary(Cov_bss)
```

    ## Subset selection object
    ## Call: regsubsets.formula(Over.Under ~ . - PTS - X - GmSc, data = train_2, 
    ##     nvmax = 7)
    ## 25 Variables  (and intercept)
    ##                Forced in Forced out
    ## Win_Loss           FALSE      FALSE
    ## FG                 FALSE      FALSE
    ## FGA                FALSE      FALSE
    ## FG_Percent         FALSE      FALSE
    ## X3P                FALSE      FALSE
    ## Differential       FALSE      FALSE
    ## X3PA               FALSE      FALSE
    ## X3P_Percent        FALSE      FALSE
    ## FT                 FALSE      FALSE
    ## FTA                FALSE      FALSE
    ## FT_Percent         FALSE      FALSE
    ## ORB                FALSE      FALSE
    ## DRB                FALSE      FALSE
    ## AST                FALSE      FALSE
    ## STL                FALSE      FALSE
    ## BLK                FALSE      FALSE
    ## TOV                FALSE      FALSE
    ## PF                 FALSE      FALSE
    ## Plus_Minus         FALSE      FALSE
    ## Postseason         FALSE      FALSE
    ## Opp_Def_Eff        FALSE      FALSE
    ## Opp_TS             FALSE      FALSE
    ## Opp_TO_per_POS     FALSE      FALSE
    ## Opp_3PT_Made       FALSE      FALSE
    ## TRB                FALSE      FALSE
    ## 1 subsets of each size up to 8
    ## Selection Algorithm: exhaustive
    ##          Win_Loss FG  FGA FG_Percent X3P Differential X3PA X3P_Percent FT  FTA
    ## 1  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         " " " "
    ## 2  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         " " " "
    ## 3  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         " " " "
    ## 4  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         " " " "
    ## 5  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         "*" " "
    ## 6  ( 1 ) " "      "*" " " " "        " " " "          " "  " "         "*" " "
    ## 7  ( 1 ) " "      "*" " " " "        "*" " "          " "  " "         "*" " "
    ## 8  ( 1 ) " "      "*" " " "*"        " " " "          " "  "*"         "*" " "
    ##          FT_Percent ORB DRB TRB AST STL BLK TOV PF  Plus_Minus Postseason
    ## 1  ( 1 ) " "        " " " " " " " " " " " " " " " " " "        " "       
    ## 2  ( 1 ) " "        " " " " " " " " "*" " " " " " " " "        " "       
    ## 3  ( 1 ) " "        " " " " " " " " "*" " " " " " " " "        " "       
    ## 4  ( 1 ) " "        " " " " " " " " "*" " " " " " " " "        " "       
    ## 5  ( 1 ) " "        " " " " " " " " "*" " " " " " " " "        " "       
    ## 6  ( 1 ) " "        "*" " " " " " " "*" " " " " " " " "        " "       
    ## 7  ( 1 ) " "        " " " " " " " " "*" " " " " " " " "        "*"       
    ## 8  ( 1 ) " "        "*" " " " " " " "*" " " " " " " " "        " "       
    ##          Opp_Def_Eff Opp_TS Opp_TO_per_POS Opp_3PT_Made
    ## 1  ( 1 ) " "         " "    " "            " "         
    ## 2  ( 1 ) " "         " "    " "            " "         
    ## 3  ( 1 ) " "         "*"    " "            " "         
    ## 4  ( 1 ) " "         "*"    " "            "*"         
    ## 5  ( 1 ) " "         "*"    " "            "*"         
    ## 6  ( 1 ) " "         "*"    " "            "*"         
    ## 7  ( 1 ) " "         "*"    " "            "*"         
    ## 8  ( 1 ) " "         "*"    " "            "*"

#### Analysis

One important takeaway from feature selection is seeing how much more
important the opponent defensive stats are in predicting the Over/Under
for Covington. This makes sense because role players can usually be shut
down by good defensive teams where Damian Lillard is not effected by
opponent defensive stats as much since he is one of the top offensive
players in the league and is capable of scoring even against the best
defensive teams.

### Logistic Regression

``` r
lg <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P + Opp_3PT_Made
            + X3P_Percent, data=train_2, method = "glm", family="binomial",
            trControl=ctrl, maxit = 100)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
print(lg)
```

    ## Generalized Linear Model 
    ## 
    ## 126 samples
    ##   9 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 100, 101, 101, 101, 101 
    ## Resampling results:
    ## 
    ##   Accuracy  Kappa    
    ##   0.968     0.9334393

``` r
#let's train the final model on the entire training set and test set 

lg_final=glm(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P + Opp_3PT_Made
            + X3P_Percent, data=train_2, family=binomial, maxit = 100)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
#check the deviance drop off 

anova(lg_final, test="Chisq")
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

    ## Analysis of Deviance Table
    ## 
    ## Model: binomial, link: logit
    ## 
    ## Response: Over.Under
    ## 
    ## Terms added sequentially (first to last)
    ## 
    ## 
    ##              Df Deviance Resid. Df Resid. Dev  Pr(>Chi)    
    ## NULL                           125    173.114              
    ## FG            1  129.994       124     43.121 < 2.2e-16 ***
    ## FTA           1   10.024       123     33.097  0.001545 ** 
    ## FT            1   16.813       122     16.284 4.125e-05 ***
    ## FG_Percent    1    0.396       121     15.888  0.529232    
    ## X3PA          1    8.356       120      7.531  0.003843 ** 
    ## FGA           1    0.782       119      6.749  0.376481    
    ## X3P           1    6.749       118      0.000  0.009379 ** 
    ## Opp_3PT_Made  1    0.000       117      0.000  0.999999    
    ## X3P_Percent   1    0.000       116      0.000  1.000000    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
log.probs <- predict(lg_final, test_2, type="response")

predicted = ifelse(log.probs > 0.5,'Over','Under')
predicted = as.factor(predicted)
confusionMatrix(test_2$Over.Under,predicted)
```

    ## Warning in confusionMatrix.default(test_2$Over.Under, predicted): Levels are not
    ## in the same order for reference and data. Refactoring data to match.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Over Under
    ##      Over    16     0
    ##      Under    0    14
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.8843, 1)
    ##     No Information Rate : 0.5333     
    ##     P-Value [Acc > NIR] : 6.456e-09  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ##                                      
    ##             Sensitivity : 1.0000     
    ##             Specificity : 1.0000     
    ##          Pos Pred Value : 1.0000     
    ##          Neg Pred Value : 1.0000     
    ##              Prevalence : 0.5333     
    ##          Detection Rate : 0.5333     
    ##    Detection Prevalence : 0.5333     
    ##       Balanced Accuracy : 1.0000     
    ##                                      
    ##        'Positive' Class : Over       
    ## 

#### Analysis

-   Logistic regression model performs perfectly on the data for Robert
    Covington.

-   Test set accuracy of 1.

### Support Vector Machine

``` r
svc <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P + Opp_3PT_Made
            + X3P_Percent,data=train_2, method="svmLinear",
            preProcess = c("center","scale"),trControl=ctrl)

print(svc)
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 126 samples
    ##   9 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## Pre-processing: centered (9), scaled (9) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 101, 101, 101, 100, 101 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9446154  0.8873462
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1

``` r
#let's perform grid search to find the optimal value for C

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 
                          1.75, 2,5))

svm_grid <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P + Opp_3PT_Made
            + X3P_Percent, data = train_2, method = "svmLinear", tuneGrid = grid,
            preProcess = c("center", "scale"),trControl=ctrl)
```

    ## Warning: model fit failed for Fold1: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold2: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold3: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold4: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning: model fit failed for Fold5: C=0.00 Error in .local(x, ...) : 
    ##   No Support Vectors found. You may want to change your parameters

    ## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo, :
    ## There were missing values in resampled performance measures.

    ## Warning in train.default(x, y, weights = w, ...): missing values found in
    ## aggregated results

``` r
svm_grid$finalModel
```

    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: C-svc  (classification) 
    ##  parameter : cost C = 1.75 
    ## 
    ## Linear (vanilla) kernel function. 
    ## 
    ## Number of Support Vectors : 20 
    ## 
    ## Objective Function Value : -19.0217 
    ## Training error : 0.02381 
    ## Probability model included.

``` r
plot(svm_grid)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

``` r
#let's train the final model on the entire training set and test set

final_grid <- expand.grid(C = c(0.75))

svm_final <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P + Opp_3PT_Made
            + X3P_Percent, data = train_2, method = "svmLinear", tuneGrid = final_grid,
            preProcess = c("center", "scale"))


SVM_pred <- predict(svm_final, newdata = test_2)
confusionMatrix(test_2$Over.Under,SVM_pred)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    14    0
    ##      Over      1   15
    ##                                           
    ##                Accuracy : 0.9667          
    ##                  95% CI : (0.8278, 0.9992)
    ##     No Information Rate : 0.5             
    ##     P-Value [Acc > NIR] : 2.887e-08       
    ##                                           
    ##                   Kappa : 0.9333          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9333          
    ##             Specificity : 1.0000          
    ##          Pos Pred Value : 1.0000          
    ##          Neg Pred Value : 0.9375          
    ##              Prevalence : 0.5000          
    ##          Detection Rate : 0.4667          
    ##    Detection Prevalence : 0.4667          
    ##       Balanced Accuracy : 0.9667          
    ##                                           
    ##        'Positive' Class : Under           
    ## 

#### Analysis

-   Grid search chose the optimal value for C at 0.75.

-   Test set accuracy of 0.9667, performing slightly worse than logistic
    regression model.

### Decision Tree

``` r
dt <- train(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P + Opp_3PT_Made
            + X3P_Percent,data=train_2, method="rpart",trControl=ctrl)

print(dt)
```

    ## CART 
    ## 
    ## 126 samples
    ##   9 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 101, 100, 101, 101, 101 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa    
    ##   0.00000000  0.9209231  0.8389129
    ##   0.02678571  0.9289231  0.8546829
    ##   0.83928571  0.6889231  0.3035469
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.02678571.

``` r
dt_pred <- predict(dt, newdata = test_2)
confusionMatrix(test_2$Over.Under,dt_pred)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    13    1
    ##      Over      1   15
    ##                                           
    ##                Accuracy : 0.9333          
    ##                  95% CI : (0.7793, 0.9918)
    ##     No Information Rate : 0.5333          
    ##     P-Value [Acc > NIR] : 2.326e-06       
    ##                                           
    ##                   Kappa : 0.8661          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9286          
    ##             Specificity : 0.9375          
    ##          Pos Pred Value : 0.9286          
    ##          Neg Pred Value : 0.9375          
    ##              Prevalence : 0.4667          
    ##          Detection Rate : 0.4333          
    ##    Detection Prevalence : 0.4667          
    ##       Balanced Accuracy : 0.9330          
    ##                                           
    ##        'Positive' Class : Under           
    ## 

``` r
#let's plot the tree

Cov.tree=tree(Over.Under~FG +  FTA + FT + FG_Percent + X3PA + FGA + X3P + Opp_3PT_Made
            + X3P_Percent,train_2)

plot(Cov.tree)
text(Cov.tree, pretty=0)
```

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->
#### Analysis

-   Test set accuracy of 0.9333

-   Covington’s tree is slightly different than Damian Lillard’s tree.
    Here we see that the first split is at FG 3.5 where Dame was at FG
    8.5. Dame’s second splits involve FT way more than Covington which
    tells us that Damian Lillard relies on getting foul calls to score
    his points. Covington relies on free throws less than Dame and we
    see that hitting some 3 pointers is more of a factor for him.

-   This makes sense because Robert Covington is a 3 and D player, which
    means his game revolves around playing good defense, getting steals,
    etc. and hitting 3 pointers off a catch and shoot.

### Stochastic Gradient Boost

``` r
sgb <- train(Over.Under~.- PTS - X - GmSc,data=train_2,method="gbm",trControl=ctrl)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2762             nan     0.1000    0.0319
    ##      2        1.1710             nan     0.1000    0.0558
    ##      3        1.0626             nan     0.1000    0.0431
    ##      4        0.9856             nan     0.1000    0.0329
    ##      5        0.9179             nan     0.1000    0.0285
    ##      6        0.8595             nan     0.1000    0.0287
    ##      7        0.8068             nan     0.1000    0.0206
    ##      8        0.7634             nan     0.1000    0.0208
    ##      9        0.7238             nan     0.1000    0.0168
    ##     10        0.6995             nan     0.1000    0.0086
    ##     20        0.5046             nan     0.1000    0.0018
    ##     40        0.3109             nan     0.1000   -0.0027
    ##     60        0.2190             nan     0.1000   -0.0013
    ##     80        0.1604             nan     0.1000   -0.0018
    ##    100        0.1215             nan     0.1000   -0.0018
    ##    120        0.0956             nan     0.1000   -0.0005
    ##    140        0.0782             nan     0.1000   -0.0017
    ##    150        0.0665             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2266             nan     0.1000    0.0709
    ##      2        1.1191             nan     0.1000    0.0535
    ##      3        1.0277             nan     0.1000    0.0488
    ##      4        0.9430             nan     0.1000    0.0403
    ##      5        0.8858             nan     0.1000    0.0164
    ##      6        0.8181             nan     0.1000    0.0296
    ##      7        0.7713             nan     0.1000    0.0119
    ##      8        0.7271             nan     0.1000    0.0163
    ##      9        0.6905             nan     0.1000    0.0127
    ##     10        0.6455             nan     0.1000    0.0148
    ##     20        0.3966             nan     0.1000    0.0049
    ##     40        0.1956             nan     0.1000    0.0009
    ##     60        0.1083             nan     0.1000   -0.0001
    ##     80        0.0685             nan     0.1000   -0.0018
    ##    100        0.0454             nan     0.1000   -0.0003
    ##    120        0.0293             nan     0.1000   -0.0005
    ##    140        0.0197             nan     0.1000   -0.0003
    ##    150        0.0161             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2320             nan     0.1000    0.0623
    ##      2        1.1255             nan     0.1000    0.0515
    ##      3        1.0258             nan     0.1000    0.0378
    ##      4        0.9603             nan     0.1000    0.0311
    ##      5        0.8855             nan     0.1000    0.0409
    ##      6        0.8178             nan     0.1000    0.0304
    ##      7        0.7665             nan     0.1000    0.0162
    ##      8        0.7199             nan     0.1000    0.0189
    ##      9        0.6692             nan     0.1000    0.0188
    ##     10        0.6315             nan     0.1000    0.0157
    ##     20        0.3845             nan     0.1000    0.0037
    ##     40        0.1786             nan     0.1000   -0.0004
    ##     60        0.0944             nan     0.1000    0.0010
    ##     80        0.0525             nan     0.1000   -0.0012
    ##    100        0.0311             nan     0.1000    0.0000
    ##    120        0.0190             nan     0.1000   -0.0001
    ##    140        0.0134             nan     0.1000   -0.0001
    ##    150        0.0109             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2154             nan     0.1000    0.0829
    ##      2        1.0930             nan     0.1000    0.0624
    ##      3        0.9825             nan     0.1000    0.0509
    ##      4        0.9021             nan     0.1000    0.0442
    ##      5        0.8408             nan     0.1000    0.0282
    ##      6        0.7862             nan     0.1000    0.0213
    ##      7        0.7234             nan     0.1000    0.0304
    ##      8        0.6716             nan     0.1000    0.0277
    ##      9        0.6373             nan     0.1000    0.0138
    ##     10        0.5901             nan     0.1000    0.0205
    ##     20        0.3574             nan     0.1000   -0.0012
    ##     40        0.2174             nan     0.1000   -0.0040
    ##     60        0.1365             nan     0.1000   -0.0001
    ##     80        0.0977             nan     0.1000    0.0006
    ##    100        0.0712             nan     0.1000   -0.0001
    ##    120        0.0541             nan     0.1000   -0.0010
    ##    140        0.0381             nan     0.1000    0.0001
    ##    150        0.0332             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2118             nan     0.1000    0.0829
    ##      2        1.0850             nan     0.1000    0.0630
    ##      3        0.9789             nan     0.1000    0.0529
    ##      4        0.8857             nan     0.1000    0.0454
    ##      5        0.8087             nan     0.1000    0.0381
    ##      6        0.7381             nan     0.1000    0.0366
    ##      7        0.6794             nan     0.1000    0.0276
    ##      8        0.6272             nan     0.1000    0.0233
    ##      9        0.5809             nan     0.1000    0.0219
    ##     10        0.5393             nan     0.1000    0.0193
    ##     20        0.3074             nan     0.1000    0.0046
    ##     40        0.1291             nan     0.1000    0.0021
    ##     60        0.0874             nan     0.1000   -0.0023
    ##     80        0.0495             nan     0.1000   -0.0012
    ##    100        0.0354             nan     0.1000   -0.0012
    ##    120        0.0223             nan     0.1000   -0.0005
    ##    140        0.0179             nan     0.1000    0.0000
    ##    150        0.0182             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2115             nan     0.1000    0.0813
    ##      2        1.0801             nan     0.1000    0.0648
    ##      3        0.9700             nan     0.1000    0.0558
    ##      4        0.8810             nan     0.1000    0.0417
    ##      5        0.8010             nan     0.1000    0.0367
    ##      6        0.7319             nan     0.1000    0.0350
    ##      7        0.6722             nan     0.1000    0.0308
    ##      8        0.6231             nan     0.1000    0.0235
    ##      9        0.5799             nan     0.1000    0.0172
    ##     10        0.5451             nan     0.1000    0.0127
    ##     20        0.2982             nan     0.1000    0.0007
    ##     40        0.1132             nan     0.1000    0.0021
    ##     60        0.0589             nan     0.1000   -0.0000
    ##     80        0.0354             nan     0.1000   -0.0008
    ##    100        0.0218             nan     0.1000   -0.0001
    ##    120        0.0149             nan     0.1000   -0.0005
    ##    140        0.0105             nan     0.1000    0.0001
    ##    150        0.0094             nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2300             nan     0.1000    0.0749
    ##      2        1.1030             nan     0.1000    0.0559
    ##      3        1.0429             nan     0.1000    0.0228
    ##      4        0.9516             nan     0.1000    0.0418
    ##      5        0.8812             nan     0.1000    0.0356
    ##      6        0.8193             nan     0.1000    0.0295
    ##      7        0.7636             nan     0.1000    0.0248
    ##      8        0.7139             nan     0.1000    0.0194
    ##      9        0.6826             nan     0.1000    0.0109
    ##     10        0.6479             nan     0.1000    0.0159
    ##     20        0.4522             nan     0.1000    0.0039
    ##     40        0.2836             nan     0.1000   -0.0009
    ##     60        0.2060             nan     0.1000   -0.0000
    ##     80        0.1437             nan     0.1000    0.0007
    ##    100        0.1023             nan     0.1000   -0.0003
    ##    120        0.0756             nan     0.1000   -0.0004
    ##    140        0.0573             nan     0.1000   -0.0002
    ##    150        0.0512             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2249             nan     0.1000    0.0725
    ##      2        1.1051             nan     0.1000    0.0620
    ##      3        1.0127             nan     0.1000    0.0498
    ##      4        0.9276             nan     0.1000    0.0383
    ##      5        0.8634             nan     0.1000    0.0319
    ##      6        0.8013             nan     0.1000    0.0303
    ##      7        0.7458             nan     0.1000    0.0245
    ##      8        0.6955             nan     0.1000    0.0142
    ##      9        0.6498             nan     0.1000    0.0137
    ##     10        0.6145             nan     0.1000    0.0169
    ##     20        0.3833             nan     0.1000    0.0026
    ##     40        0.1880             nan     0.1000    0.0007
    ##     60        0.0985             nan     0.1000   -0.0001
    ##     80        0.0564             nan     0.1000   -0.0006
    ##    100        0.0333             nan     0.1000   -0.0015
    ##    120        0.0209             nan     0.1000   -0.0003
    ##    140        0.0123             nan     0.1000   -0.0002
    ##    150        0.0099             nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2242             nan     0.1000    0.0697
    ##      2        1.1087             nan     0.1000    0.0579
    ##      3        1.0114             nan     0.1000    0.0451
    ##      4        0.9223             nan     0.1000    0.0355
    ##      5        0.8603             nan     0.1000    0.0299
    ##      6        0.8029             nan     0.1000    0.0247
    ##      7        0.7496             nan     0.1000    0.0127
    ##      8        0.7045             nan     0.1000    0.0253
    ##      9        0.6624             nan     0.1000    0.0203
    ##     10        0.6267             nan     0.1000    0.0125
    ##     20        0.3754             nan     0.1000    0.0062
    ##     40        0.1664             nan     0.1000   -0.0007
    ##     60        0.0744             nan     0.1000   -0.0005
    ##     80        0.0459             nan     0.1000   -0.0001
    ##    100        0.0258             nan     0.1000   -0.0000
    ##    120        0.0153             nan     0.1000    0.0002
    ##    140        0.0094             nan     0.1000    0.0001
    ##    150        0.0071             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2364             nan     0.1000    0.0738
    ##      2        1.1196             nan     0.1000    0.0582
    ##      3        1.0132             nan     0.1000    0.0447
    ##      4        0.9347             nan     0.1000    0.0384
    ##      5        0.8725             nan     0.1000    0.0325
    ##      6        0.8082             nan     0.1000    0.0249
    ##      7        0.7539             nan     0.1000    0.0208
    ##      8        0.7145             nan     0.1000    0.0103
    ##      9        0.6766             nan     0.1000    0.0214
    ##     10        0.6426             nan     0.1000    0.0163
    ##     20        0.4499             nan     0.1000    0.0014
    ##     40        0.2870             nan     0.1000   -0.0014
    ##     60        0.2009             nan     0.1000   -0.0010
    ##     80        0.1412             nan     0.1000   -0.0017
    ##    100        0.1081             nan     0.1000   -0.0009
    ##    120        0.0829             nan     0.1000   -0.0015
    ##    140        0.0630             nan     0.1000    0.0001
    ##    150        0.0547             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2234             nan     0.1000    0.0678
    ##      2        1.0985             nan     0.1000    0.0592
    ##      3        1.0030             nan     0.1000    0.0458
    ##      4        0.9279             nan     0.1000    0.0370
    ##      5        0.8521             nan     0.1000    0.0341
    ##      6        0.7934             nan     0.1000    0.0281
    ##      7        0.7440             nan     0.1000    0.0228
    ##      8        0.6823             nan     0.1000    0.0245
    ##      9        0.6451             nan     0.1000    0.0162
    ##     10        0.5963             nan     0.1000    0.0165
    ##     20        0.3654             nan     0.1000    0.0017
    ##     40        0.1771             nan     0.1000    0.0010
    ##     60        0.0951             nan     0.1000   -0.0011
    ##     80        0.0573             nan     0.1000    0.0000
    ##    100        0.0342             nan     0.1000   -0.0003
    ##    120        0.0228             nan     0.1000   -0.0000
    ##    140        0.0159             nan     0.1000   -0.0002
    ##    150        0.0123             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2285             nan     0.1000    0.0734
    ##      2        1.1065             nan     0.1000    0.0564
    ##      3        1.0128             nan     0.1000    0.0459
    ##      4        0.9277             nan     0.1000    0.0440
    ##      5        0.8629             nan     0.1000    0.0304
    ##      6        0.8161             nan     0.1000    0.0141
    ##      7        0.7567             nan     0.1000    0.0270
    ##      8        0.7027             nan     0.1000    0.0237
    ##      9        0.6526             nan     0.1000    0.0221
    ##     10        0.6126             nan     0.1000    0.0098
    ##     20        0.3690             nan     0.1000    0.0010
    ##     40        0.1607             nan     0.1000   -0.0012
    ##     60        0.0807             nan     0.1000   -0.0000
    ##     80        0.0434             nan     0.1000    0.0005
    ##    100        0.0232             nan     0.1000   -0.0000
    ##    120        0.0144             nan     0.1000   -0.0002
    ##    140        0.0082             nan     0.1000    0.0001
    ##    150        0.0063             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2402             nan     0.1000    0.0675
    ##      2        1.1461             nan     0.1000    0.0440
    ##      3        1.0461             nan     0.1000    0.0436
    ##      4        0.9876             nan     0.1000    0.0240
    ##      5        0.9197             nan     0.1000    0.0345
    ##      6        0.8516             nan     0.1000    0.0299
    ##      7        0.8003             nan     0.1000    0.0260
    ##      8        0.7532             nan     0.1000    0.0211
    ##      9        0.7090             nan     0.1000    0.0175
    ##     10        0.6690             nan     0.1000    0.0152
    ##     20        0.4605             nan     0.1000   -0.0021
    ##     40        0.3014             nan     0.1000   -0.0004
    ##     60        0.2185             nan     0.1000   -0.0010
    ##     80        0.1679             nan     0.1000    0.0004
    ##    100        0.1382             nan     0.1000    0.0002
    ##    120        0.1051             nan     0.1000   -0.0006
    ##    140        0.0875             nan     0.1000   -0.0012
    ##    150        0.0778             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2395             nan     0.1000    0.0680
    ##      2        1.1303             nan     0.1000    0.0448
    ##      3        1.0348             nan     0.1000    0.0397
    ##      4        0.9528             nan     0.1000    0.0343
    ##      5        0.8797             nan     0.1000    0.0328
    ##      6        0.8269             nan     0.1000    0.0179
    ##      7        0.7697             nan     0.1000    0.0284
    ##      8        0.7116             nan     0.1000    0.0187
    ##      9        0.6626             nan     0.1000    0.0231
    ##     10        0.6249             nan     0.1000    0.0169
    ##     20        0.3903             nan     0.1000    0.0062
    ##     40        0.1973             nan     0.1000    0.0018
    ##     60        0.1137             nan     0.1000   -0.0004
    ##     80        0.0704             nan     0.1000   -0.0009
    ##    100        0.0410             nan     0.1000   -0.0002
    ##    120        0.0290             nan     0.1000   -0.0005
    ##    140        0.0205             nan     0.1000   -0.0003
    ##    150        0.0181             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2245             nan     0.1000    0.0737
    ##      2        1.1177             nan     0.1000    0.0579
    ##      3        1.0233             nan     0.1000    0.0417
    ##      4        0.9500             nan     0.1000    0.0343
    ##      5        0.8713             nan     0.1000    0.0360
    ##      6        0.8091             nan     0.1000    0.0245
    ##      7        0.7602             nan     0.1000    0.0158
    ##      8        0.7109             nan     0.1000    0.0213
    ##      9        0.6732             nan     0.1000    0.0143
    ##     10        0.6349             nan     0.1000    0.0140
    ##     20        0.3844             nan     0.1000    0.0004
    ##     40        0.1828             nan     0.1000   -0.0008
    ##     60        0.1062             nan     0.1000   -0.0012
    ##     80        0.0598             nan     0.1000   -0.0011
    ##    100        0.0350             nan     0.1000   -0.0001
    ##    120        0.0216             nan     0.1000   -0.0006
    ##    140        0.0131             nan     0.1000    0.0001
    ##    150        0.0102             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2382             nan     0.1000    0.0639
    ##      2        1.1321             nan     0.1000    0.0583
    ##      3        1.0441             nan     0.1000    0.0453
    ##      4        0.9730             nan     0.1000    0.0337
    ##      5        0.8919             nan     0.1000    0.0358
    ##      6        0.8284             nan     0.1000    0.0294
    ##      7        0.7772             nan     0.1000    0.0252
    ##      8        0.7314             nan     0.1000    0.0204
    ##      9        0.6961             nan     0.1000    0.0154
    ##     10        0.6690             nan     0.1000    0.0114
    ##     20        0.4569             nan     0.1000    0.0004
    ##     40        0.2867             nan     0.1000    0.0003
    ##     60        0.2141             nan     0.1000   -0.0031
    ##     80        0.1555             nan     0.1000   -0.0027
    ##    100        0.1269             nan     0.1000   -0.0009

``` r
print(sgb)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 126 samples
    ##  28 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 101, 101, 100, 101, 101 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.9366154  0.8706271
    ##   1                  100      0.9446154  0.8873306
    ##   1                  150      0.9366154  0.8709372
    ##   2                   50      0.9289231  0.8546796
    ##   2                  100      0.9366154  0.8706271
    ##   2                  150      0.9289231  0.8546796
    ##   3                   50      0.9366154  0.8709527
    ##   3                  100      0.9366154  0.8712690
    ##   3                  150      0.9366154  0.8712690
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 100, interaction.depth =
    ##  1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
#let's train the final model on the entire training set and test set 

final_grid <- expand.grid(n.trees = c(50), 
                          interaction.depth = c(3),
                          shrinkage = c(0.1), n.minobsinnode = 10)

sgb_final <- train(Over.Under~.- PTS - X - GmSc - Win_Loss,data=train_2,method="gbm",
                     tuneGrid = final_grid)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2123             nan     0.1000    0.0809
    ##      2        1.0762             nan     0.1000    0.0699
    ##      3        0.9586             nan     0.1000    0.0564
    ##      4        0.8626             nan     0.1000    0.0468
    ##      5        0.7768             nan     0.1000    0.0412
    ##      6        0.7042             nan     0.1000    0.0346
    ##      7        0.6453             nan     0.1000    0.0284
    ##      8        0.5920             nan     0.1000    0.0257
    ##      9        0.5457             nan     0.1000    0.0221
    ##     10        0.5019             nan     0.1000    0.0219
    ##     20        0.2402             nan     0.1000    0.0087
    ##     40        0.0825             nan     0.1000    0.0001
    ##     50        0.0506             nan     0.1000    0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1154             nan     0.1000    0.0733
    ##      2        1.0026             nan     0.1000    0.0455
    ##      3        0.9163             nan     0.1000    0.0387
    ##      4        0.8366             nan     0.1000    0.0373
    ##      5        0.7697             nan     0.1000    0.0320
    ##      6        0.7124             nan     0.1000    0.0254
    ##      7        0.6644             nan     0.1000    0.0264
    ##      8        0.6237             nan     0.1000    0.0161
    ##      9        0.5857             nan     0.1000    0.0125
    ##     10        0.5475             nan     0.1000    0.0173
    ##     20        0.3024             nan     0.1000    0.0035
    ##     40        0.1129             nan     0.1000    0.0028
    ##     50        0.0756             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1829             nan     0.1000    0.0765
    ##      2        1.0667             nan     0.1000    0.0571
    ##      3        0.9903             nan     0.1000    0.0368
    ##      4        0.9054             nan     0.1000    0.0430
    ##      5        0.8374             nan     0.1000    0.0349
    ##      6        0.7805             nan     0.1000    0.0270
    ##      7        0.7292             nan     0.1000    0.0224
    ##      8        0.6825             nan     0.1000    0.0208
    ##      9        0.6448             nan     0.1000    0.0157
    ##     10        0.5985             nan     0.1000    0.0174
    ##     20        0.3464             nan     0.1000    0.0009
    ##     40        0.1365             nan     0.1000    0.0018
    ##     50        0.1009             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2372             nan     0.1000    0.0674
    ##      2        1.1194             nan     0.1000    0.0533
    ##      3        1.0235             nan     0.1000    0.0448
    ##      4        0.9292             nan     0.1000    0.0456
    ##      5        0.8559             nan     0.1000    0.0302
    ##      6        0.7902             nan     0.1000    0.0309
    ##      7        0.7367             nan     0.1000    0.0216
    ##      8        0.6950             nan     0.1000    0.0150
    ##      9        0.6366             nan     0.1000    0.0249
    ##     10        0.6012             nan     0.1000    0.0121
    ##     20        0.3050             nan     0.1000    0.0044
    ##     40        0.1137             nan     0.1000    0.0018
    ##     50        0.0740             nan     0.1000    0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1578             nan     0.1000    0.0742
    ##      2        1.0332             nan     0.1000    0.0586
    ##      3        0.9237             nan     0.1000    0.0510
    ##      4        0.8448             nan     0.1000    0.0324
    ##      5        0.7633             nan     0.1000    0.0356
    ##      6        0.6924             nan     0.1000    0.0338
    ##      7        0.6340             nan     0.1000    0.0293
    ##      8        0.5835             nan     0.1000    0.0204
    ##      9        0.5380             nan     0.1000    0.0238
    ##     10        0.4995             nan     0.1000    0.0193
    ##     20        0.2591             nan     0.1000    0.0062
    ##     40        0.0780             nan     0.1000    0.0014
    ##     50        0.0500             nan     0.1000   -0.0006
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2417             nan     0.1000    0.0737
    ##      2        1.1136             nan     0.1000    0.0630
    ##      3        1.0048             nan     0.1000    0.0524
    ##      4        0.9183             nan     0.1000    0.0440
    ##      5        0.8382             nan     0.1000    0.0388
    ##      6        0.7701             nan     0.1000    0.0307
    ##      7        0.7143             nan     0.1000    0.0291
    ##      8        0.6703             nan     0.1000    0.0137
    ##      9        0.6309             nan     0.1000    0.0115
    ##     10        0.5901             nan     0.1000    0.0163
    ##     20        0.3338             nan     0.1000    0.0005
    ##     40        0.1266             nan     0.1000    0.0009
    ##     50        0.0866             nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2375             nan     0.1000    0.0749
    ##      2        1.1200             nan     0.1000    0.0601
    ##      3        1.0318             nan     0.1000    0.0410
    ##      4        0.9514             nan     0.1000    0.0385
    ##      5        0.8759             nan     0.1000    0.0319
    ##      6        0.8141             nan     0.1000    0.0302
    ##      7        0.7579             nan     0.1000    0.0194
    ##      8        0.7050             nan     0.1000    0.0240
    ##      9        0.6659             nan     0.1000    0.0147
    ##     10        0.6321             nan     0.1000    0.0123
    ##     20        0.3352             nan     0.1000    0.0051
    ##     40        0.1280             nan     0.1000    0.0010
    ##     50        0.0915             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2100             nan     0.1000    0.0828
    ##      2        1.0695             nan     0.1000    0.0658
    ##      3        0.9577             nan     0.1000    0.0583
    ##      4        0.8701             nan     0.1000    0.0389
    ##      5        0.7960             nan     0.1000    0.0328
    ##      6        0.7404             nan     0.1000    0.0179
    ##      7        0.6717             nan     0.1000    0.0375
    ##      8        0.6122             nan     0.1000    0.0285
    ##      9        0.5608             nan     0.1000    0.0252
    ##     10        0.5129             nan     0.1000    0.0221
    ##     20        0.2581             nan     0.1000    0.0035
    ##     40        0.0844             nan     0.1000   -0.0005
    ##     50        0.0535             nan     0.1000    0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2285             nan     0.1000    0.0699
    ##      2        1.1123             nan     0.1000    0.0549
    ##      3        1.0197             nan     0.1000    0.0443
    ##      4        0.9367             nan     0.1000    0.0401
    ##      5        0.8626             nan     0.1000    0.0312
    ##      6        0.7972             nan     0.1000    0.0322
    ##      7        0.7450             nan     0.1000    0.0251
    ##      8        0.6949             nan     0.1000    0.0201
    ##      9        0.6543             nan     0.1000    0.0171
    ##     10        0.6179             nan     0.1000    0.0080
    ##     20        0.3593             nan     0.1000    0.0049
    ##     40        0.1403             nan     0.1000    0.0006
    ##     50        0.0981             nan     0.1000    0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2244             nan     0.1000    0.0819
    ##      2        1.0995             nan     0.1000    0.0656
    ##      3        0.9938             nan     0.1000    0.0514
    ##      4        0.9020             nan     0.1000    0.0415
    ##      5        0.8225             nan     0.1000    0.0376
    ##      6        0.7567             nan     0.1000    0.0276
    ##      7        0.6978             nan     0.1000    0.0278
    ##      8        0.6455             nan     0.1000    0.0237
    ##      9        0.5996             nan     0.1000    0.0219
    ##     10        0.5563             nan     0.1000    0.0148
    ##     20        0.3074             nan     0.1000    0.0051
    ##     40        0.1006             nan     0.1000    0.0029
    ##     50        0.0640             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2422             nan     0.1000    0.0653
    ##      2        1.1242             nan     0.1000    0.0551
    ##      3        1.0262             nan     0.1000    0.0490
    ##      4        0.9414             nan     0.1000    0.0351
    ##      5        0.8666             nan     0.1000    0.0346
    ##      6        0.8032             nan     0.1000    0.0283
    ##      7        0.7576             nan     0.1000    0.0187
    ##      8        0.7043             nan     0.1000    0.0254
    ##      9        0.6453             nan     0.1000    0.0274
    ##     10        0.6030             nan     0.1000    0.0210
    ##     20        0.3235             nan     0.1000    0.0050
    ##     40        0.1285             nan     0.1000    0.0026
    ##     50        0.0863             nan     0.1000    0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2197             nan     0.1000    0.0576
    ##      2        1.0964             nan     0.1000    0.0523
    ##      3        0.9946             nan     0.1000    0.0485
    ##      4        0.9138             nan     0.1000    0.0341
    ##      5        0.8443             nan     0.1000    0.0297
    ##      6        0.7745             nan     0.1000    0.0300
    ##      7        0.7134             nan     0.1000    0.0286
    ##      8        0.6630             nan     0.1000    0.0195
    ##      9        0.6257             nan     0.1000    0.0128
    ##     10        0.5864             nan     0.1000    0.0125
    ##     20        0.2932             nan     0.1000    0.0038
    ##     40        0.1056             nan     0.1000   -0.0004
    ##     50        0.0610             nan     0.1000    0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2307             nan     0.1000    0.0672
    ##      2        1.1072             nan     0.1000    0.0576
    ##      3        1.0094             nan     0.1000    0.0471
    ##      4        0.9286             nan     0.1000    0.0408
    ##      5        0.8561             nan     0.1000    0.0361
    ##      6        0.8067             nan     0.1000    0.0151
    ##      7        0.7561             nan     0.1000    0.0174
    ##      8        0.7023             nan     0.1000    0.0232
    ##      9        0.6545             nan     0.1000    0.0183
    ##     10        0.6158             nan     0.1000    0.0162
    ##     20        0.3379             nan     0.1000    0.0037
    ##     40        0.1213             nan     0.1000    0.0018
    ##     50        0.0756             nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1497             nan     0.1000    0.0776
    ##      2        1.0293             nan     0.1000    0.0573
    ##      3        0.9403             nan     0.1000    0.0426
    ##      4        0.8531             nan     0.1000    0.0362
    ##      5        0.7936             nan     0.1000    0.0265
    ##      6        0.7339             nan     0.1000    0.0255
    ##      7        0.6881             nan     0.1000    0.0203
    ##      8        0.6409             nan     0.1000    0.0222
    ##      9        0.6032             nan     0.1000    0.0126
    ##     10        0.5559             nan     0.1000    0.0196
    ##     20        0.3000             nan     0.1000    0.0063
    ##     40        0.1054             nan     0.1000   -0.0001
    ##     50        0.0688             nan     0.1000    0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2474             nan     0.1000    0.0635
    ##      2        1.1453             nan     0.1000    0.0560
    ##      3        1.0481             nan     0.1000    0.0452
    ##      4        0.9707             nan     0.1000    0.0376
    ##      5        0.9017             nan     0.1000    0.0276
    ##      6        0.8472             nan     0.1000    0.0167
    ##      7        0.7968             nan     0.1000    0.0223
    ##      8        0.7503             nan     0.1000    0.0174
    ##      9        0.7097             nan     0.1000    0.0156
    ##     10        0.6700             nan     0.1000    0.0176
    ##     20        0.3767             nan     0.1000    0.0048
    ##     40        0.1477             nan     0.1000    0.0025
    ##     50        0.0987             nan     0.1000    0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1458             nan     0.1000    0.0697
    ##      2        1.0405             nan     0.1000    0.0596
    ##      3        0.9520             nan     0.1000    0.0368
    ##      4        0.8776             nan     0.1000    0.0384
    ##      5        0.8128             nan     0.1000    0.0288
    ##      6        0.7609             nan     0.1000    0.0179
    ##      7        0.7117             nan     0.1000    0.0224
    ##      8        0.6731             nan     0.1000    0.0133
    ##      9        0.6397             nan     0.1000    0.0127
    ##     10        0.6068             nan     0.1000    0.0115
    ##     20        0.3620             nan     0.1000    0.0086
    ##     40        0.1452             nan     0.1000   -0.0000
    ##     50        0.0963             nan     0.1000    0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2128             nan     0.1000    0.0686
    ##      2        1.0896             nan     0.1000    0.0598
    ##      3        0.9846             nan     0.1000    0.0527
    ##      4        0.8952             nan     0.1000    0.0436
    ##      5        0.8195             nan     0.1000    0.0359
    ##      6        0.7497             nan     0.1000    0.0315
    ##      7        0.6940             nan     0.1000    0.0266
    ##      8        0.6461             nan     0.1000    0.0243
    ##      9        0.6026             nan     0.1000    0.0200
    ##     10        0.5549             nan     0.1000    0.0147
    ##     20        0.2903             nan     0.1000    0.0051
    ##     40        0.1100             nan     0.1000   -0.0011
    ##     50        0.0693             nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2199             nan     0.1000    0.0791
    ##      2        1.0937             nan     0.1000    0.0647
    ##      3        0.9851             nan     0.1000    0.0526
    ##      4        0.8966             nan     0.1000    0.0413
    ##      5        0.8320             nan     0.1000    0.0282
    ##      6        0.7693             nan     0.1000    0.0300
    ##      7        0.7084             nan     0.1000    0.0272
    ##      8        0.6556             nan     0.1000    0.0184
    ##      9        0.6064             nan     0.1000    0.0233
    ##     10        0.5686             nan     0.1000    0.0144
    ##     20        0.3238             nan     0.1000    0.0033
    ##     40        0.1325             nan     0.1000   -0.0026
    ##     50        0.0933             nan     0.1000   -0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1835             nan     0.1000    0.0690
    ##      2        1.0680             nan     0.1000    0.0558
    ##      3        0.9714             nan     0.1000    0.0483
    ##      4        0.8898             nan     0.1000    0.0401
    ##      5        0.8180             nan     0.1000    0.0273
    ##      6        0.7524             nan     0.1000    0.0278
    ##      7        0.6885             nan     0.1000    0.0241
    ##      8        0.6345             nan     0.1000    0.0201
    ##      9        0.5924             nan     0.1000    0.0169
    ##     10        0.5616             nan     0.1000    0.0118
    ##     20        0.3046             nan     0.1000    0.0060
    ##     40        0.1062             nan     0.1000    0.0028
    ##     50        0.0624             nan     0.1000    0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2583             nan     0.1000    0.0583
    ##      2        1.1454             nan     0.1000    0.0557
    ##      3        1.0523             nan     0.1000    0.0430
    ##      4        0.9714             nan     0.1000    0.0394
    ##      5        0.8962             nan     0.1000    0.0325
    ##      6        0.8472             nan     0.1000    0.0191
    ##      7        0.7872             nan     0.1000    0.0240
    ##      8        0.7373             nan     0.1000    0.0206
    ##      9        0.6925             nan     0.1000    0.0165
    ##     10        0.6466             nan     0.1000    0.0205
    ##     20        0.3741             nan     0.1000    0.0058
    ##     40        0.1529             nan     0.1000    0.0019
    ##     50        0.1021             nan     0.1000    0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2132             nan     0.1000    0.0735
    ##      2        1.0966             nan     0.1000    0.0569
    ##      3        1.0091             nan     0.1000    0.0416
    ##      4        0.9192             nan     0.1000    0.0439
    ##      5        0.8503             nan     0.1000    0.0324
    ##      6        0.7865             nan     0.1000    0.0288
    ##      7        0.7322             nan     0.1000    0.0209
    ##      8        0.6739             nan     0.1000    0.0237
    ##      9        0.6283             nan     0.1000    0.0198
    ##     10        0.5906             nan     0.1000    0.0180
    ##     20        0.3172             nan     0.1000    0.0063
    ##     40        0.1042             nan     0.1000    0.0016
    ##     50        0.0660             nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2405             nan     0.1000    0.0699
    ##      2        1.1247             nan     0.1000    0.0586
    ##      3        1.0253             nan     0.1000    0.0453
    ##      4        0.9386             nan     0.1000    0.0400
    ##      5        0.8665             nan     0.1000    0.0317
    ##      6        0.8080             nan     0.1000    0.0213
    ##      7        0.7478             nan     0.1000    0.0300
    ##      8        0.6962             nan     0.1000    0.0251
    ##      9        0.6504             nan     0.1000    0.0160
    ##     10        0.6108             nan     0.1000    0.0172
    ##     20        0.3540             nan     0.1000    0.0054
    ##     40        0.1278             nan     0.1000   -0.0004
    ##     50        0.0816             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2399             nan     0.1000    0.0731
    ##      2        1.1133             nan     0.1000    0.0561
    ##      3        1.0168             nan     0.1000    0.0424
    ##      4        0.9300             nan     0.1000    0.0384
    ##      5        0.8567             nan     0.1000    0.0284
    ##      6        0.7912             nan     0.1000    0.0293
    ##      7        0.7272             nan     0.1000    0.0311
    ##      8        0.6668             nan     0.1000    0.0246
    ##      9        0.6186             nan     0.1000    0.0238
    ##     10        0.5733             nan     0.1000    0.0210
    ##     20        0.3143             nan     0.1000    0.0103
    ##     40        0.1307             nan     0.1000   -0.0004
    ##     50        0.0820             nan     0.1000    0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2119             nan     0.1000    0.0706
    ##      2        1.0992             nan     0.1000    0.0572
    ##      3        1.0028             nan     0.1000    0.0461
    ##      4        0.9212             nan     0.1000    0.0381
    ##      5        0.8546             nan     0.1000    0.0316
    ##      6        0.7933             nan     0.1000    0.0261
    ##      7        0.7382             nan     0.1000    0.0286
    ##      8        0.6893             nan     0.1000    0.0211
    ##      9        0.6415             nan     0.1000    0.0185
    ##     10        0.6024             nan     0.1000    0.0182
    ##     20        0.3596             nan     0.1000    0.0020
    ##     40        0.1367             nan     0.1000    0.0021
    ##     50        0.0919             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2370             nan     0.1000    0.0699
    ##      2        1.1195             nan     0.1000    0.0451
    ##      3        1.0211             nan     0.1000    0.0509
    ##      4        0.9306             nan     0.1000    0.0419
    ##      5        0.8583             nan     0.1000    0.0346
    ##      6        0.7900             nan     0.1000    0.0328
    ##      7        0.7320             nan     0.1000    0.0200
    ##      8        0.6774             nan     0.1000    0.0257
    ##      9        0.6302             nan     0.1000    0.0205
    ##     10        0.5855             nan     0.1000    0.0194
    ##     20        0.3296             nan     0.1000    0.0055
    ##     40        0.1189             nan     0.1000    0.0021
    ##     50        0.0794             nan     0.1000    0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2311             nan     0.1000    0.0731
    ##      2        1.1092             nan     0.1000    0.0593
    ##      3        1.0085             nan     0.1000    0.0460
    ##      4        0.9193             nan     0.1000    0.0437
    ##      5        0.8461             nan     0.1000    0.0335
    ##      6        0.7956             nan     0.1000    0.0193
    ##      7        0.7357             nan     0.1000    0.0266
    ##      8        0.6828             nan     0.1000    0.0222
    ##      9        0.6377             nan     0.1000    0.0141
    ##     10        0.6055             nan     0.1000    0.0147
    ##     20        0.3543             nan     0.1000    0.0006
    ##     40        0.1547             nan     0.1000   -0.0013
    ##     50        0.1081             nan     0.1000   -0.0004

``` r
sgb.predict <- predict(sgb_final, newdata = test_2)
confusionMatrix(test_2$Over.Under,sgb.predict)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    13    1
    ##      Over      1   15
    ##                                           
    ##                Accuracy : 0.9333          
    ##                  95% CI : (0.7793, 0.9918)
    ##     No Information Rate : 0.5333          
    ##     P-Value [Acc > NIR] : 2.326e-06       
    ##                                           
    ##                   Kappa : 0.8661          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9286          
    ##             Specificity : 0.9375          
    ##          Pos Pred Value : 0.9286          
    ##          Neg Pred Value : 0.9375          
    ##              Prevalence : 0.4667          
    ##          Detection Rate : 0.4333          
    ##    Detection Prevalence : 0.4667          
    ##       Balanced Accuracy : 0.9330          
    ##                                           
    ##        'Positive' Class : Under           
    ## 

``` r
sgb_final$finalModel
```

    ## A gradient boosted model with bernoulli loss function.
    ## 50 iterations were performed.
    ## There were 24 predictors of which 21 had non-zero influence.

#### Analysis

-   test set accuracy of 0.933.

-   The SGB model did not improve the decision tree. Let’s see what a
    bagging technique like random forest will do.

### Random Forest

``` r
rf <- train(Over.Under~.- PTS - X - GmSc - Win_Loss,data=train_2,method="rf",trControl=ctrl)

print(rf)
```

    ## Random Forest 
    ## 
    ## 126 samples
    ##  28 predictor
    ##   2 classes: 'Under', 'Over' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 101, 101, 100, 101, 101 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9123077  0.8197982
    ##   13    0.9283077  0.8526130
    ##   24    0.9283077  0.8522967
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 13.

``` r
#let's perform grid search to find more optimal values for 'mtry'

tunegrid <- expand.grid(.mtry = (1:15)) 

rf_grid <- train(Over.Under ~.- PTS - X - GmSc - Win_Loss, data = train_2, method = 'rf',
                   tuneGrid = tunegrid,trControl=ctrl)

rf_grid$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x))) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 4
    ## 
    ##         OOB estimate of  error rate: 7.14%
    ## Confusion matrix:
    ##       Under Over class.error
    ## Under    50    6  0.10714286
    ## Over      3   67  0.04285714

``` r
#let's train the final model on the entire training set and test set 

rf_final <- randomForest(Over.Under ~ .- PTS - X - GmSc - Win_Loss, data = train_2, mtry = 13)

rf.predict <- predict(rf_final, test_2, type = "response")
confusionMatrix(test_2$Over.Under,rf.predict)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Under Over
    ##      Under    13    1
    ##      Over      1   15
    ##                                           
    ##                Accuracy : 0.9333          
    ##                  95% CI : (0.7793, 0.9918)
    ##     No Information Rate : 0.5333          
    ##     P-Value [Acc > NIR] : 2.326e-06       
    ##                                           
    ##                   Kappa : 0.8661          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9286          
    ##             Specificity : 0.9375          
    ##          Pos Pred Value : 0.9286          
    ##          Neg Pred Value : 0.9375          
    ##              Prevalence : 0.4667          
    ##          Detection Rate : 0.4333          
    ##    Detection Prevalence : 0.4667          
    ##       Balanced Accuracy : 0.9330          
    ##                                           
    ##        'Positive' Class : Under           
    ## 

#### Analysis

-   test set accuracy of 0.9667

-   The random forest model did improve our model by introducing some
    randomness.

``` r
#let's plot ROC curves of the Decision Tree, SGB and Random Forest to evaluate 
#how the bagging and boosting techniques improved model performance

roc_plots <- evalm(list(dt,sgb,rf_grid),gnames=c('Decison Tree','Stochastic GB',
                                                 'Random Forest'))
```

    ## ***MLeval: Machine Learning Model Evaluation***

    ## Input: caret train function object

    ## Not averaging probs.

    ## Group 1 type: cv

    ## Group 2 type: cv

    ## Group 3 type: cv

    ## Observations: 378

    ## Number of groups: 3

    ## Observations per group: 126

    ## Positive: Over

    ## Negative: Under

    ## Group: Decison Tree

    ## Positive: 70

    ## Negative: 56

    ## Group: Stochastic GB

    ## Positive: 70

    ## Negative: 56

    ## Group: Random Forest

    ## Positive: 70

    ## Negative: 56

    ## ***Performance Metrics***

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-33-1.png)<!-- -->![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-33-2.png)<!-- -->![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-33-3.png)<!-- -->

    ## Decison Tree Optimal Informedness = 0.846428571428572

    ## Stochastic GB Optimal Informedness = 0.903571428571428

    ## Random Forest Optimal Informedness = 0.885714285714286

    ## Decison Tree AUC-ROC = 0.89

    ## Stochastic GB AUC-ROC = 0.98

    ## Random Forest AUC-ROC = 0.98

![](Over_Under_Classification_files/figure-gfm/unnamed-chunk-33-4.png)<!-- -->
### Limitations

-   it’s important to keep in mind that we have a small dataset that we
    are working with here. In my opinion, increasing the number of game
    logs should be done before deploying any model similar to this one,
    even with cross-validation.

Realistically, it is impossible to use features such as FG or X3P to
make predictions because we don’t have the values until after the game
is played.

In order to be profitable in sports betting, we need to have a
prediction accuracy of at least 55% since the books price bets in a way
where the house always make money.

However, we do have variables like Opp_3PT_Made and Opp_TS available
before the game is played and if we are able to get many opponent
defensive stats compiled into the spreadsheet, we can create an ensemble
that turns these weak learners into a strong learner.

The issue is that this data collection process is extremely time
consuming which is why this dataset only has 4 opponent defensive stats
but if anyone does have the time and creates a profitable model,
definitely use it to your advantage! It would be a worthwhile
experiment.

#### Example:

``` r
lg_ex=glm(Over.Under~Opp_TS + Opp_3PT_Made + Opp_TO_per_POS + Opp_Def_Eff, data=train_2, family=binomial, maxit = 100)

#check the deviance drop off 

anova(lg_ex, test="Chisq")
```

    ## Analysis of Deviance Table
    ## 
    ## Model: binomial, link: logit
    ## 
    ## Response: Over.Under
    ## 
    ## Terms added sequentially (first to last)
    ## 
    ## 
    ##                Df Deviance Resid. Df Resid. Dev Pr(>Chi)
    ## NULL                             125     173.11         
    ## Opp_TS          1  0.11318       124     173.00   0.7366
    ## Opp_3PT_Made    1  2.13128       123     170.87   0.1443
    ## Opp_TO_per_POS  1  0.28096       122     170.59   0.5961
    ## Opp_Def_Eff     1  2.64846       121     167.94   0.1037

``` r
log.probs <- predict(lg_ex, test_2, type="response")

predicted = ifelse(log.probs > 0.5,'Over','Under')
predicted = as.factor(predicted)
confusionMatrix(test_2$Over.Under,predicted)
```

    ## Warning in confusionMatrix.default(test_2$Over.Under, predicted): Levels are not
    ## in the same order for reference and data. Refactoring data to match.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Over Under
    ##      Over    11     5
    ##      Under   11     3
    ##                                           
    ##                Accuracy : 0.4667          
    ##                  95% CI : (0.2834, 0.6567)
    ##     No Information Rate : 0.7333          
    ##     P-Value [Acc > NIR] : 0.9995          
    ##                                           
    ##                   Kappa : -0.1009         
    ##                                           
    ##  Mcnemar's Test P-Value : 0.2113          
    ##                                           
    ##             Sensitivity : 0.5000          
    ##             Specificity : 0.3750          
    ##          Pos Pred Value : 0.6875          
    ##          Neg Pred Value : 0.2143          
    ##              Prevalence : 0.7333          
    ##          Detection Rate : 0.3667          
    ##    Detection Prevalence : 0.5333          
    ##       Balanced Accuracy : 0.4375          
    ##                                           
    ##        'Positive' Class : Over            
    ## 

The logistic model has an accuracy of 0.4667 on the test set with 4
opponent defensive stats. It’s very possible that creating new variables
using feature combination or just adding tons of opponent defensive
stats from the web (very time-consuming!!!) can potentially increase
this model’s accuracy to over 0.55.
