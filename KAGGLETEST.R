
train <- read.csv("/Users/Mac/Downloads/train.csv")
test  <- read.csv("/Users/Mac/Downloads/test.csv")

set.seed(5678)

library(GPfit)
library(reshape2)
library(scales)
library(dplyr)
library(lhs)
library(RMySQL)
library(xgboost)
library(Matrix)

BayesianOptimization.MySQL <- setRefClass(
  Class = "BayesianOptimization.MySQL",
  fields = list(
    f = "function",
    X = "data.frame",
    output = "data.frame",
    bounds = "list",
    scaled.X = "matrix",
    nug.thres = "numeric",
    gaussian.process = "ANY",
    pred.c = "list",
    ei.c = "numeric",
    scaled.pt.c = "numeric",
    con = "ANY"
  )
)

BayesianOptimization.MySQL$methods(
  initialize = function(X = data.frame(NULL), output = data.frame(NULL), 
                        bounds = list(), f = function(){}, connection = NULL){
    .self$con <- connection
    .self$bounds <- bounds
    .self$f <- f
    xgbsmall <- update.from.sql(con)
    .self$X <- rbind(xgbsmall[names(hyper.bounds)], X)
    .self$output <- rbind(xgbsmall["test.auc.mean"], output)
  },
  scaleToBounds = function(hypercube, bounds){
    mapply(function(x, y) rescale(x, to = y, from = c(0, 1)), hypercube, bounds)
  },
  randomSample = function(samples = 20, ...){
    if(samples > 1){
      rmat <- matrix(runif(length(bounds) * samples), samples)
      rdat <- data.frame(rmat)
      sample.points <- scaleToBounds(rdat, bounds)
      sample.points <- data.frame(sample.points)
      if(!is.null(names(bounds)))
        colnames(sample.points) <- names(bounds)
      .self$X <- rbind(X, sample.points)
      for(i in 1:samples)
        .self$output <- rbind(output, .self$f(sample.points[i,]))
    }
    else
      "Variable 'samples' must be more than 1."
  },
  fit = function(nug.thres = 20, ...){
    .self$nug.thres <- nug.thres
    .self$scaled.X <- mapply(function(x, y) rescale(x, from = y), X, bounds)
    .self$gaussian.process <- 
      GP_fit(scaled.X, output[[1]], nug_thres = nug.thres,...)
  },
  predict = function(partitions = 1000){
    hypercube <- randomLHS(partitions, length(bounds))
    predict.GP(gaussian.process, hypercube)
  },
  expectedImprovement = function(pred, ymax, epsilon = 0){
    sigma <- sqrt(pred$MSE)
    z <- (pred$Y_hat - ymax - epsilon)/sigma
    ei <- (pred$Y_hat - ymax - epsilon)*pnorm(z) + sigma*dnorm(z)
    ei[sigma == 0] <- 0
    ei
  },
  oneIter = function(partitions = 1000, nug.thres = 20, epsilon = 0, ...){
    fit(nug.thres, ...)
    .self$pred.c <- predict(partitions)
    .self$ei.c <- expectedImprovement(pred.c, max(output), epsilon)
    ex.indices <- colnames(pred.c$complete_data) %in% c("Y_hat", "MSE")
    scaled.pt <- pred.c$complete_data[which.max(ei.c), !ex.indices]
    names(scaled.pt) <- colnames(scaled.X)
    scaled.pt
  },
  run = function(iterations = 10, partitions = 1000, nug.thres = 20,
                 epsilon = 0, plot = FALSE, ...){
    if (dim(X) == c(0, 0) || dim(output) == c(0, 0))
      randomSample(...)
    for(i in 1:iterations){
      
      if(length(scaled.pt.c)){
        x.next <- scaleToBounds(scaled.pt.c, bounds)
        print(x.next)
        .self$f(data.frame(as.list(x.next)))
        xgbsmall <- update.from.sql(con)
        .self$X <- xgbsmall[names(hyper.bounds)]
        .self$output <- xgbsmall["test.auc.mean"]
      }
      .self$scaled.pt.c <- oneIter(partitions, nug.thres, epsilon, ...)
    }
  },
  resample = function(subsample = 3, repetitions = 4){
    # tests 'subsample' points again 'repetition' times
    sub.X <- sample_n(X, subsample)
    for(i in 1:repetitions){
      for(n in 1:subsample){
        y <- f(sub.X[n,])
        .self$output <- rbind(output, setNames(y, names(output)))
      }
    }
    .self$X <- do.call(
      "rbind", c(list(X), rep(list(sub.X), repetitions), make.row.names = FALSE))
  },
  update.from.sql = function(con){
    xgbout <- dbReadTable(con, "XGBOUT")
    for(i in names(hyper.bounds)){
      xgbout <- filter(xgbout, 
                       hyper.bounds[[i]][1] <= xgbout[i],
                       xgbout[i] <= hyper.bounds[[i]][2])
    }
    xgbsmall.opt <- xgbout %>% 
      group_by(eta, max_depth = round(max_depth), subsample, 
               colsample_bytree, lambda, alpha, min_child_weight) %>% 
      summarise(nrounds = round(nrounds[which.max(test.auc.mean)]), 
                test.auc.mean = max(test.auc.mean))
    xgbsmall.max <- xgbout %>%
      group_by(eta, max_depth = round(max_depth), subsample, 
               colsample_bytree, lambda, alpha, min_child_weight) %>% 
      summarise(nrounds = round(max(nrounds)), 
                test.auc.mean = test.auc.mean[which.max(nrounds)])
    xgbsmall.min <- xgbout %>%
      group_by(eta, max_depth = round(max_depth), subsample, 
               colsample_bytree, lambda, alpha, min_child_weight) %>% 
      summarise(nrounds = round(min(nrounds)), 
                test.auc.mean = test.auc.mean[which.min(nrounds)])
    xgbsmall <- bind_rows(xgbsmall.opt, xgbsmall.max, xgbsmall.min)
    data.frame(xgbsmall)
  }
)

# Script below forked from Kaggle open source

##### Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

##### 0 count per line
count0 <- function(x) {
  return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN=count0)
test$n0 <- apply(test, 1, FUN=count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]

#---limit vars in test based on min and max vals of train

print('Setting min-max lims on test data')
for(f in colnames(train)){
  lim <- min(train[,f])
  test[test[,f]<lim,f] <- lim
  
  lim <- max(train[,f])
  test[test[,f]>lim,f] <- lim
}

#---

train$TARGET <- train.y


train <- sparse.model.matrix(TARGET ~ ., data = train)

dtrain <- xgb.DMatrix(data=train, label=train.y)
watchlist <- list(train=dtrain)

boost.hyper <- function(x){
  
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "auc",
                  eta                 = x$eta,
                  max_depth           = round(x$max_depth),
                  subsample           = x$subsample,
                  colsample_bytree    = x$colsample_bytree,
                  lambda              = x$lambda,
                  alpha               = x$alpha,
                  min_child_weight    = x$min_child_weight
  )
  
  clf <- xgb.cv(   params              = param, 
                   data                = dtrain, 
                   nrounds             = round(x$nrounds),
                   nfold               = 4,
                   verbose             = 1,
                   metrics             = "auc",
                   #early.stop.round    = 65,
                   watchlist           = watchlist,
                   maximize            = FALSE
  )
  x.nonrounds<- x[,-match("nrounds", names(x))]
  data.xgb <- cbind(x.nonrounds, clf, nrounds = 1:nrow(clf))
  dbWriteTable(con, "XGBOUT", data.xgb, append = TRUE)
  return(max(clf$test.auc.mean))
}

### Hyper parameter tuning using baysian optimization of kaggle open source 
### gradient boosting algorithm.
con <- dbConnect(MySQL(),"XGBOUT", username = "root", password = "0000")
hyper.bounds <- list(eta = c(0, .07), max_depth = c(1,7), 
                     subsample = c(0.6, 1), colsample_bytree = c(0.6, 1), 
                     nrounds = c(200, 1500), lambda = c(0.4,1.5), alpha = c(0,1.5),
                     min_child_weight = c(1, 2))
BO <- BayesianOptimization.MySQL(f=boost.hyper, bounds = hyper.bounds, connection = con)
BO$run(iterations = 3, partitions = 50000, nug.thres = 20)
