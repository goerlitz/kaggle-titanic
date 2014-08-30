## setwd("~/study/coursera/datasci/kaggle-titanic")

library(rpart)         # decision tree
library(randomForest)
library(e1071)         # svm

data <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

# need factors
data$Survived <- as.factor(data$Survived)
for (i in c("Pclass","SibSp","Parch")) {
  data[,i] <- as.factor(data[,i])
  test[,i] <- as.factor(test[,i])
}
# with identical levels
levels(data$Parch) <- levels(test$Parch) <- seq(0,9)
levels(data$Embarked) <- levels(test$Embarked) <- c("", "C", "Q", "S")

# replace NA in Age/Fare with reasonable default value (0, mean?)
data$Age[is.na(data$Age)] <- mean(data$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] <- mean(data$Age, na.rm = TRUE)
test$Fare[is.na(test$Fare)] <- mean(data$Fare, na.rm = TRUE)

# add cabin letter
data$CabinLetter <- as.factor(gsub(" .*|[0-9]", "", data$Cabin))
test$CabinLetter <- as.factor(gsub(" .*|[0-9]", "", test$Cabin))
levels(data$CabinLetter) <- levels(test$CabinLetter) <- c("", LETTERS)

# group fare in intervals
data$FareGroup <- cut(data$Fare, breaks=hist(as.numeric(data$Fare), breaks="Scott", plot=FALSE)$breaks)
fareQuantile <- quantile(data$Fare, probs=seq(0,1,0.1))
data$FareEqui <- cut(data$Fare, breaks=hist(as.numeric(data$Fare), breaks=fareQuantile, plot=FALSE)$breaks)

# compare survival rate with other factors
table(data$Survived, data$Sex)
table(data$Survived, data$CabinLetter)
table(data$Survived, data$Pclass)
table(data$Survived, data$Age)
table(data$Survived, data$Embarked)
table(data$Survived, data$Parch) # parent, child
table(data$Survived, data$SibSp) # sibling, spouse
table(data$Survived, data$FareGroup)


withTree <- function(fol) {
  writeLines(paste("training decision tree:", deparse(fol)))
  treeModel <- rpart(fol, method="class", data=data)
#  print(treeModel)
  treePred <- predict(treeModel, newdata=test, type = "class")
  print(table(treePred))
  treePred
}

withForest <- function(fol) {
  writeLines(paste("training random forest:", deparse(fol)))
  forestModel <- randomForest(fol, data=data)
  print(forestModel)
  # print(importance(forestModel))
  # CAUTION: cannot predict with missing levels/ NA values -> data/test must have same level
  forestPred <- predict(forestModel, newdata=test, type = "class")
  print(table(forestPred))
  forestPred
}

withSVM <- function(fol) {
  writeLines(paste("training svm:", deparse(fol)))
  svmModel <- svm(fol, data=data)
  svmPred <- predict(svmModel, newdata=test, type = "class")
  print(table(svmPred))
  svmPred
}

# define factor combinations to be used for model learning
formulas <- c(
  formula(Survived ~ Sex + Pclass + Age),
  formula(Survived ~ Sex + Pclass + Fare),
  formula(Survived ~ Sex + Pclass + Age + Fare),
  formula(Survived ~ Sex + Pclass + Age + Parch + SibSp),
  formula(Survived ~ Sex + Pclass + Age + CabinLetter),
  formula(Survived ~ Sex + Pclass + Age + CabinLetter + Fare),
  formula(Survived ~ Pclass + CabinLetter + Fare),
  formula(Survived ~ Sex + Pclass + Age + CabinLetter + Embarked),
  formula(Survived ~ Sex + Pclass + Age + Parch + SibSp + CabinLetter + Embarked),
  formula(Survived ~ Sex + Pclass + Age + Parch + SibSp + CabinLetter + Embarked + Fare),
  formula(Survived ~ Sex + Age + CabinLetter + Fare),
  formula(Survived ~ Sex + Age + Fare)
)

# training decision trees
for (f in formulas) withTree(f)

# training random forests
for (f in formulas) withForest(f)

# training svm
for (f in formulas) withSVM(f)

export <- function(filename, result) {
  result <- cbind(test$PassengerId, as.data.frame(result))
  names(result) <- c("PassengerId", "Survived")
  result$Survived <- as.numeric(result$Survived) -1
  write.table(result, filename, sep=",", row.names=FALSE)
}

#export("predictionSVM.csv", withSVM(formula(Survived ~ Sex + Age + Fare))) # score: 0.77033
#export("predictionSVM.csv", withSVM(formula(Survived ~ Sex + Pclass + Age + Fare))) # score: 0.77512
#export("predictionSVM.csv", withSVM(formula(Survived ~ Sex + Pclass + Age + CabinLetter + Fare))) # score: 0.76555
export("predictionForest.csv", withForest(formula(Survived ~ Sex + Pclass + Age + Fare))) # score: 0.77990
