library(GPfit)
library(reshape2)
library(scales)
library(dplyr)
library(lhs)

BayesianOptimization <- setRefClass(
  Class = "BayesianOptimization",
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
    scaled.pt.c = "numeric"
    )
  )

BayesianOptimization$methods(
  initialize = function(X = data.frame(NULL), 
                        output = data.frame(NULL), 
                        bounds = list(),
                        f = function(){}){
    .self$X <- X
    .self$output <- output
    .self$bounds <- bounds
    .self$f <- f
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
        .self$X <- rbind(X, x.next)
        cat("iteration-", i, "next attempt -", x.next, fill = TRUE)
        y.new <- .self$f(data.frame(as.list(x.next)))
        cat("new output -", as.numeric(y.new), fill = TRUE)
        .self$output <- rbind(output, y.new)
        if(plot)
          choosePlot(scaled.pt.c)
      }
      .self$scaled.pt.c <- oneIter(partitions, nug.thres, epsilon, ...)
    }
    cat("x* :", X[order(BO$output) == 1,], "output* :", max(output), fill =TRUE)
  },
  nextIter = function(next.x = NULL, next.output = NULL, 
                      partitions = 1000, nug.thres = 20, 
                      epsilon = 0, plot = FALSE, ...){
    if (dim(X) == c(0, 0) || dim(output) == c(0, 0))
      randomSample(...)
    if(!is.null(next.x) && !is.null(next.output)){
      .self$X <- rbind(X, next.x)
      .self$output <- rbind(output, next.output)
    }
    scaled.pt <- oneIter(partitions, nug.thres, epsilon, ...)
    if(plot) 
      choosePlot(scaled.pt)
    scaleToBounds(scaled.pt, bounds)
  },
  scaleToBounds = function(hypercube, bounds){
    mapply(function(x, y) rescale(x, to = y, from = c(0, 1)), hypercube, bounds)
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
  choosePlot = function(scaled.pt){
      if(length(scaled.pt) == 1)
        oneDimPlot(pred.c, ei.c, scaled.pt)
      else if(length(scaled.pt) == 2)
        twoDimPlot(pred.c, ei.c, scaled.pt)
      else
        warning("Cannot plot bayesian optimization in more than 2 dimensions.")
  },
  oneDimPlot = function(pred, ei, scaled.pt){
    par(mfrow=c(2,1), oma = c(0,0,2,0), mar = c(2,4,0,0))
    plot(gaussian.process)
    abline(v = scaled.pt, col = "green", lwd = 2)
    
    ex.indices <- colnames(pred$complete_data) %in% c("Y_hat", "MSE")
    ei.df <- data.frame(pred$complete_data[,!ex.indices], ei)
    ei.df <- ei.df[order(ei.df[names(ei.df) != "ei"]),]
    ei.zero <- data.frame(0, 0)
    names(ei.zero) <- names(ei.df)
    ei.one <- data.frame(1, 0)
    names(ei.one) <- names(ei.df)
    ei.df <- bind_rows(ei.zero, ei.df, ei.one)
    
    plot(ei.df, type = 'l', xlab ="")
    polygon(ei.df[[which(names(ei.df) != "ei")]], ei.df$ei, col = "orange")
    abline(v = scaled.pt, col = "green", lwd = 1.5)
    mtext("Bayesian Optimiztion 1D", outer = TRUE, cex = 1.5)
  },
  twoDimPlot = function(pred, ei, scaled.pt){
    plot(gaussian.process)
  },
  animated.run = function(...){
    plot <- TRUE
    saveGIF(run(plot = plot, ...), movie.name = "animation.gif")
  }
)
