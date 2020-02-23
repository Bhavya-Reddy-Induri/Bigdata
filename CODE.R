#install.packages("rpart.plot")
#install.packages("neuralnet")
#install.packages("e1071")
#install.packages("h2o")
#install.packages("deepnet")
#install.packages("class")
#install.packages("adabag")
#install.packages("randomForest")
#install.packages("gbm")
#install.packages("xgboost")
#install.packages("mlbench")
#install.packages("caret")
#install.packages("ipred")
#install.packages("lattice")
#install.packages("pROC")
#install.packages("rpart")
library("pROC")
library("lattice")
library("ipred")
library("caret")
library("mlbench")
library("xgboost")
library("gbm")
library("randomForest")
library("adabag")
library("e1071")
library("class")
library("neuralnet")
library("rpart")
library("rpart.plot")
library("h2o")
library("deepnet")
transfusion <- read.csv("/Users/pradeepkumar/Downloads/data.csv",header = TRUE)
maxs = apply(transfusion, MARGIN = 2, max)
mins = apply(transfusion, MARGIN = 2, min)
scaled_set = as.data.frame(scale(transfusion, center = mins, scale = maxs-mins))
#scaled_set$class <- factor(scaled_set$class)
correlation_matrix<-cor(scaled_set[,1:5])
print(correlation_matrix)
highly_correlated<-findCorrelation(correlation_matrix, cutoff = 0.95)
print(highly_correlated)
scaled_set<-scaled_set[c(1,3,4,5)]
n_folds<-5

#######DECISION TREE########

acc<-c()
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  fit <- rpart(class ~ Recency + Time + Monetary , data = train_data, method = "class", parms = list(split="information"))
  predict <- predict(fit,test_data,type="class")
  roc<-roc(test_data$class,as.numeric(predict))
  accuracy = mean(test_data$class == predict)
  acc<-c(acc,accuracy)
}
cat("Accuracy of Decision Tree :",mean(acc),"\n")
cat("\n")
print("ROC for Decision tree")
print(roc)

##########PERCEPTRON###############
acc1<-c()
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  nn<- neuralnet(class ~ Recency + Time + Monetary , data = train_data,hidden = 0, threshold = 0.1,err.fct = "sse", linear.output = FALSE,act.fct = "logistic")
  predictions<-compute(nn,test_data[,1:3])$net.result
  predictions<-ifelse(predictions>1,1,0)
  accuracy1 = mean(test_data$class == predictions)
  acc1<-c(acc1,accuracy1)
  roc1<-roc(test_data$class,as.numeric(predictions))
}
cat("Accuracy of Perceptron :",mean(acc1),"\n")
cat("\n")
print("ROC for perceptron:")
roc11<-roc
print(roc1)
#######NEURALNETS###########

acc2<-c()
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  nn<- neuralnet(class ~ Recency + Time + Monetary , data = train_data,hidden = c(2,2), err.fct = "ce",rep = 5, threshold = 0.01,linear.output = FALSE, learningrate = 0.5,act.fct = "logistic")
  predictions<-compute(nn,test_data[,1:3])$net.result
  predictions<-ifelse(predictions>1,1,0)
  accuracy2 = mean(test_data$class == predictions)
  acc2<-c(acc2,accuracy2)
  roc2<-roc(test_data$class,as.numeric(predictions))
}
cat("Accuracy of Neural Nets:",mean(acc2),"\n")
cat("\n")
print("ROC Nueral Nets")
print(roc2)
##########SVM###########
acc3<-c()
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  svm_fit <- svm(class ~., data = train_data, kernel = "linear", cost = 100, gamma = 1)
  svmpredict<-predict(svm_fit,test_data[,1:3])
  svmpredict<-ifelse(svmpredict>0.5,1,0)
  accuracy3 = mean(test_data$class == svmpredict)
  acc3<-c(acc3,accuracy3)
  roc3<-roc(test_data$class,as.numeric(svmpredict))
}

cat("Accuracy of SVM :",mean(acc3),"\n")
cat("\n")
print("ROC for Support vector machine")
print(roc3)
#######DEEP LEARNING########
acc4<-c()
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  dnn<-dbn.dnn.train(as.matrix(train_data[,1:3]),train_data[,4],numepochs = 100, momentum = 0.5, activationfun = "sigm",learningrate_scale=1)
  dlpredict<-nn.test(dnn,as.matrix(test_data[,1:3]),test_data[,4],t=0.5)
  acc4<-c(acc4,(1-dlpredict))
}
cat("Accuracy of Deep Learning :",mean(acc4),"\n")
cat("\n")
print("ROC for Deep learning")
print(roc11)
#########NAIVE BAYES#######
acc5<-c()
scaled_set$class <- factor(scaled_set$class)
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  model<-naiveBayes(class ~., data = train_data,laplace = 3)
  pred<-predict(model,test_data[,1:3])
  pred
  tab<-table(pred,test_data$class)
  accuracy5 = sum(tab[row(tab)==col(tab)])/sum(tab)
  acc5<-c(acc5,accuracy5)
  roc4<-roc(test_data$class,as.numeric(pred))
}
cat("Accuracy of Naive Bayes :",mean(acc5),"\n")
cat("\n")
print("ROC for Naive bayes")
print(roc4)
########LOGISTIC REGRESSION#######

acc6<-c()
scaled_set$class <- factor(scaled_set$class)
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  model<-glm(class~ Recency + Time + Monetary ,data = train_data,family="binomial")
  pred<-predict.glm(model,test_data[,1:3])
  pred<-ifelse(pred>0.5,1,0) 
  accuracy6 = mean(pred==test_data$class)
  acc6<-c(acc6,accuracy6)
  roc5<-roc(test_data$class,as.numeric(pred))
}
cat("Accuracy of Logistic regression :",mean(acc6),"\n")
cat("\n")
print("ROC for Logistic regression")
print(roc5)

########KNN############

acc7<-c()
scaled_set$class <- factor(scaled_set$class)
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  model<-knn(train = train_data,test = test_data, cl=train_data$class,k=4,prob = TRUE,use.all = TRUE)
  accuracy7 = mean(model==test_data$class)
  acc7<-c(acc7,accuracy7)
  roc6<-roc(test_data$class,as.numeric(model))
}
cat("Accuracy of KNN :",mean(acc7),"\n")
cat("\n")
print("ROC for K-nearest neighbors")
print(roc6)
#######ADA BOOSTING#######

acc8<-c()
scaled_set$class <- factor(scaled_set$class)
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  model<-boosting(class~., data=train_data, mfinal = 10,control = rpart.control(maxdepth = 1))
  pred<-predict.boosting(model,test_data)
  accuracy8 = mean(pred$class==test_data$class)
  acc8<-c(acc8,accuracy8)
  roc7<-roc(test_data$class,as.numeric(pred$class))
}
cat("Accuracy of Ada Boosting :",mean(acc8),"\n")
cat("\n")
print("ROC for ADA Boosting")
print(roc7)


#######RANDOM FOREST#####

acc9<-c()
scaled_set$class <- factor(scaled_set$class)
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  model<-randomForest(class~., data=train_data,importance = TRUE, mtry = 2, ntree = 500)
  pred<-predict(model,test_data)
  accuracy9 = mean(pred==test_data$class)
  acc9<-c(acc9,accuracy9)
  roc8<-roc(test_data$class,as.numeric(pred))
}
cat("Accuracy of Random Forest :",mean(acc9),"\n")
cat("\n")
print("ROC for Random Forest")
print(roc8)
#####GRADIENT BOOSTING######

acc10<-c()
scaled_set$class <- factor(scaled_set$class)
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  model<-xgboost( data = as.matrix(train_data[,1:3]), label=train_data$class, max.depth=2,nrounds = 2)
  pred<-predict(model,as.matrix(test_data[,1:3]))
  pred<-ifelse(pred>1,1,0)
  accuracy10 = mean(pred==test_data$class)
  acc10<-c(acc10,accuracy10)
  roc9<-roc(test_data$class,as.numeric(pred))
}
cat("Accuracy of Gradient boosting :",mean(acc10),"\n")
cat("\n")
print("ROC for Gradient Boosting")
print(roc9)
######BAGGING######

acc11<-c()
for (i in 1:5)
{
  test_data <- scaled_set[(((i-1)*length(scaled_set[,1])/n_folds)+1):((i)*length(scaled_set[,1])/n_folds),]
  train_data <- rbind(scaled_set[0:(((i-1)*length(scaled_set[,1])/n_folds)),],scaled_set[(((i)*length(scaled_set[,1])/n_folds)+1):length(scaled_set[,1]),])
  test_data<- na.omit(test_data)
  train_data<-na.omit(train_data)
  model <- bagging(class~.,ns=275,nbagg=500,
                   control=rpart.control(minsplit=5, cp=0, xval=0,maxsurrogate=0),
                   data=train_data)
  pred<-predict(model,(test_data[,1:3]))
  pred<-ifelse(pred$class>0.5,1,0)
  accuracy11 = mean(pred==test_data$class)
  acc11<-c(acc11,accuracy11)
  roc10<-roc(test_data$class,as.numeric(pred))
}
cat("Accuracy of Bagging :",mean(acc11),"\n")
cat("\n")
print("ROC for Bagging")
print(roc10)
