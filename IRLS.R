setwd("C:/Users/spkgy/OneDrive/Tsinghua/ML/Homework 1")
library("Matrix")
library("caret")
library("dplyr")
library("MASS")


irls <- function(X, y, Xtest, trueClass, maxit = 100) {
  w = matrix(0, nrow = length(X[1,]))
  prev = w + 1
  n = 0
  
  while (max(abs(prev - w)) > 0.000000001 & maxit > 0) {
    prev = w
    mu = sigmoid(X %*% w)
    R = as.numeric(mu * (1 - mu))
    grad = t(X) %*% (y - mu)
    H = -(t(X) %*% (R * X))
    w = w - (solve(H) %*% grad)
    n = n + 1
    maxit = maxit - 1
    
    p = as.numeric(ifelse(X %*% w > 0, 1, 0))
    predictions = as.numeric(ifelse(Xtest %*% w > 0, 1, 0))
    
    d = data.frame("Predicted" = factor(predictions),
                   "Actual" = factor(trueClass))
    d2 = data.frame("Predicted" = factor(p),
                    "Actual" = factor(y))
    
    c = confusionMatrix(data = d$Predicted, reference = d$Actual)
    c2 = confusionMatrix(data = d2$Predicted, reference = d2$Actual)
    
    print("training data acc")
    print(c2$overall[1])
    print("test data acc")
    print(c$overall[1])
    return(c)
  }
  print(n)

}

irlsl <- function(X, y, lambda, maxit = 100) {
  w = integer(length(X[1,]))
  prev = w + 1
  n = 0
  
  while (max(abs(prev - w)) > 0.000001 & maxit > 0) {
    prev = w
    mu = as.numeric(sigmoid(X %*% w))
    R = as.numeric(mu * (1 - mu))
    grad = as.numeric(t(X) %*% (y - mu) - lambda * w)
    H = -(t(X) %*% (R * X) + lambda * diag(length(X[1,])))
    w = as.numeric(w - (solve(H) %*% grad))
    n = n + 1
    maxit = maxit - 1
  }
  print(n)
  return(w)
}

crossVal <- function(X, y, lambda, indexes) {
  train = X[indexes, ]
  ytrain = y[indexes]
  test = X[-indexes, ]
  ytest = y[-indexes]
  
  w = irlsl(train, ytrain, lambda)
  
  predictions = as.numeric(ifelse(test %*% w > 0, 1, 0))
  
  d = data.frame("Predicted" = factor(predictions),
                 "Actual" = factor(ytest))
  
  c = confusionMatrix(data = d$Predicted, reference = d$Actual)
  
  return(c$overall[1])
}

sigmoid <- function(n) {
  return(1 / (1 + exp(-n)))
}


IRLS <- function(X, y, lambda, Xtest, trueClass) {
  w = irlsl(X, y, lambda)
  
  predictions = as.numeric(ifelse(Xtest %*% w > 0, 1, 0))
  
  d = data.frame("Predicted" = factor(predictions),
                 "Actual" = factor(trueClass))
  
  c = confusionMatrix(data = d$Predicted, reference = d$Actual)
  
  print(c$overall[1])
  return(c)
}



data <- read.csv("a9a Data/a9a",
                 sep = " ",
                 header = FALSE,
                 skipNul = TRUE)

testData <- read.csv("a9a Data/a9a.t",
                     sep = " ",
                     header = FALSE,
                     skipNul = TRUE)

data <- data[, -16]
testData <- testData[, -16]

for (i in 2:length(data[1,])) {
  data[, i] = as.numeric(substr(data[, i], 1, nchar(data[, i]) - 2))
}

for (i in 2:length(testData[1,])) {
  testData[, i] = as.numeric(substr(testData[, i], 1, nchar(testData[, i]) - 2))
}

data <- data %>% mutate(V1 = ifelse(V1 == -1, 0, 1))
testData <- testData %>% mutate(V1 = ifelse(V1 == -1, 0, 1))


y <- matrix(data[, 1], nrow = length(data[, 1]))
trueClass <- matrix(testData[, 1], nrow = length(testData[, 1]))
data = data[, -1]
testData = testData[, -1]

X = matrix(0, nrow = length(data[, 1]), ncol = 123)
Xtest = matrix(0, nrow = length(testData[, 1]), ncol = 123)

for (i in 1:length(X[, 1])) {
  X[i, c(as.numeric(data[i,]))] = 1
}

for (i in 1:length(Xtest[, 1])) {
  Xtest[i, c(as.numeric(testData[i,]))] = 1
}

for (i in 1:length(X[1,])) {
  X[, i] = X[, i] - mean(X[, i])
}

for (i in 1:length(Xtest[1,])) {
  Xtest[, i] = Xtest[, i] - mean(Xtest[, i])
}


fulldata <- data.frame(X)
fulldata$y=factor(y)
fullTestData <- data.frame(Xtest)
fullTestData$y = factor(trueClass)

X = cbind(1, X)
Xtest = cbind(1, Xtest)


nonregularisedaccuracy = irls(X, y, Xtest, trueClass)


indexes <-
  fulldata$y %>% createDataPartition(p = 0.85, list = FALSE)

# for (i in 9:12) {
#   print(i)
#   print(crossVal(X, y, i, indexes))
# }

regularisedaccuracy = IRLS(X, y, 64, Xtest, trueClass)
