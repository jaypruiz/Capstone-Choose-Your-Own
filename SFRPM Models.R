if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(RCurl)) install.packages("RCurl", repos = "http://cran.us.r-project.org")

########################
# Importing and Cleaning
########################

rawdata<-getURL("https://raw.githubusercontent.com/jaypruiz/Capstone-Projects/master/rpmdata.csv")
rpmdata<-read.csv(text=rawdata,colClasses = c("character","character",rep("numeric",9),rep("factor",3)))

summary(rpmdata)

rpmdata$NAME<-NULL
rpmdata$TEAM<-NULL
rpmdata$X<-NULL
#This is an optional step. I personally did not use these columns,
#so I purged them to make it easy to read 

##############
# Partitioning
##############

sampsize <- floor(.8*nrow(rpmdata))
set.seed(3080)
#sample is a randomized process
train_indeces <- sample(1:nrow(rpmdata), size=sampsize)
trainer <- rpmdata[train_indeces,]
tester <- rpmdata[-train_indeces,]

#############################
#Testing using RPM- KNN Model
#############################

set.seed(3080)
#train is randomized
train_knn_pop <- train(POPULAR~RPM, #Testing "Popular" (Votes > 500000)
                   data=trainer,
                   trControl= trainControl(method = "repeatedcv", 
                                           number = 10, repeats = 3),
                   method="knn", 
                   preProcess = c("center", "scale"))
#It's critical to preprocess the data, especially considering our sample
train_knn_pop$results
y_hat_knn_pop <- predict(train_knn_pop, tester)
confusionMatrix(data=y_hat_knn_pop, reference = tester$POPULAR)
#evaluate the effectivity of the model

set.seed(3080)
train_knn_sal <- train(HIGH.SALARY~RPM, #Testing "High Salary" (>$7.7M)
                       data=trainer,
                       trControl= trainControl(method = "repeatedcv", 
                                               number = 10, repeats = 3),
                       method="knn", 
                       preProcess = c("center", "scale"))
train_knn_sal$results
y_hat_knn_sal <- predict(train_knn_sal, tester)
confusionMatrix(data=y_hat_knn_sal, reference = tester$POPULAR)


set.seed(3080)
train_knn_win <- train(PLAYOFFS~RPM, #Testing Winningness (Made Playoffs)
                       data=trainer,
                       trControl= trainControl(method = "repeatedcv", 
                                               number = 10, repeats = 3),
                       method="knn", 
                       preProcess = c("center", "scale"))
train_knn_win$results
y_hat_knn_win <- predict(train_knn_win, tester)
confusionMatrix(data=y_hat_knn_win, reference = tester$POPULAR)

#############################
#Testing using ORPM- KNN Model
#############################

set.seed(3080)
train_knn_orwin <- train(PLAYOFFS~ORPM, #Testing Winningness with ORPM
                       data=trainer,
                       trControl= trainControl(method = "repeatedcv", 
                                               number = 10, repeats = 3),
                       method="knn", 
                       preProcess = c("center", "scale"))
train_knn_orwin$results
y_hat_knn_orwin <- predict(train_knn_orwin, tester)
confusionMatrix(data=y_hat_knn_orwin, reference = tester$PLAYOFFS)
#lower specificity, but MUCH lower accuracy.

##############################################################
#Trying to improve RPM accuracy on Playoffs using other models
##############################################################
models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "multinom", "qda", "rf", "adaboost")

set.seed(3080)
fits <- lapply(models, function(model){ 
  print(model)
  train(PLAYOFFS~RPM, method = model, data = trainer, 
        preProcess = c("center", "scale"))
}) 

names(fits) <- models

pred <- sapply(fits, function(object) 
  predict(object, newdata = tester))
#sapply applies the anonymous function that runs predict() using all the
#models included in the object models

mean(fits$glm$results$Accuracy,
     fits$lda$results$Accuracy,
     fits$naive_bayes$results$Accuracy,
     fits$svmLinear$results$Accuracy,
     fits$knn$results$Accuracy,
     fits$gamLoess$results$Accuracy,
     fits$multinom$results$Accuracy,
     fits$qda$results$Accuracy,
     fits$rf$results$Accuracy,
     fits$adaboost$results$Accuracy)
#Total combination did not perform as well as the knn in isolation

list<-c(fits$glm$results$Accuracy,
     fits$lda$results$Accuracy,
     fits$naive_bayes$results$Accuracy,
     fits$svmLinear$results$Accuracy,
     fits$knn$results$Accuracy,
     fits$gamLoess$results$Accuracy,
     fits$multinom$results$Accuracy,
     fits$qda$results$Accuracy,
     fits$rf$results$Accuracy,
     fits$adaboost$results$Accuracy)
list
#Interestingly, with these evaluations, it appears that lda along with
#naive bayes to perform the best out of all the models.
