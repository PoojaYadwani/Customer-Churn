rm(list = ls())
setwd("C:/Users/Click/Desktop/project1_custchurn")
getwd()
# #loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "e1071",
      "DataCombine", "pROC", "doSNOW", "class", "readxl","ROSE","dplyr", "plyr", "reshape","xlsx","pbapply", "caret")

# #install.packages if not
lapply(x, install.packages)

# #load libraries
lapply(x, require, character.only = TRUE)
rm(x)


#Input Train & Test Data Source
train_Original = read.csv('Train_data.csv',header = T,na.strings = c(""," ","NA"))
test_Original = read.csv('Test_data.csv',header = T,na.strings = c(""," ","NA"))
#Creating backup of orginal data 
train = train_Original  
test = test_Original

###########################################################################
#                  EXPLORING DATA										  #
###########################################################################

#viewing the data
head(train,4)
dim(train)

#structure of data or data types
str(train)  

#Summary of data 
summary(train)

#unique value of each count
apply(train, 2,function(x) length(table(x)))

#Replacing the dot b/w collumn name to underscore for easy to use
names(train) <- gsub('\\.','_',names(train))
names(test) <- gsub('\\.','_',names(test))

#Converting area code as factor
train$area_code <- as.factor(train$area_code)
test$area_code <- as.factor(test$area_code)

#Removing phone number
train$phone_number <- NULL
test$phone_number <- NULL

#Let's see the percentage of our target variable
round(prop.table(table(train$Churn))*100,2)

# False.   True. 
# 85.51   14.49 
#Our target Class is suffering from target imbalance

#########################################################################
#          Checking Missing data										#
#########################################################################
apply(train, 2, function(x) {sum(is.na(x))}) # in R, 1 = Row & 2 = Col 
apply(test, 2, function(x) {sum(is.na(x))}) 

#No missing data found

#########################################################################
#                     Visualizing the data                              #
#########################################################################

#library(ggplot2)
#Target class distribution 
ggplot(train,aes(Churn))+
  geom_bar(colour="black", fill = "blue") +  labs(y='Churn Count', title = 'Customer Churn Statistics')

# Churning of customer Vs. State
ggplot(train, aes(fill=Churn, x=state)) +
  geom_bar(position="dodge") + labs(title="Churning ~ State")

# Churning of customer Vs. Voice Mail Plan
ggplot(train, aes(fill=Churn, x=voice_mail_plan)) +
  geom_bar(position="dodge") + labs(title="Churning ~ Voice Mail Plan")

# Churning of customer Vs. international_plan
ggplot(train, aes(fill=Churn, x=international_plan)) +
  geom_bar(position="dodge") + labs(title="Churning ~ international_plan")

# Churning of customer Vs. area_code
ggplot(train, aes(fill=Churn, x=area_code)) +
  geom_bar(position="dodge") + labs(title="Churning ~ Area Code")

# Apply the factor function for converting Categorical to level -> factors
#train <- factor(train)
#test <- factor(test)


# Verifying that conversion was successful
print(is.factor(train))
print(is.factor(test))

#all numeric var
num_index = sapply(train, is.numeric)
num_data = train[,num_index]
num_col = colnames(num_data) #storing all the column name

#Checking for categorical features
cat_index = sapply(train,is.factor) #Fetching all the categorical index & later data
cat_data = train[,cat_index]
cat_col = colnames(cat_data)[-5]  #Removing target var

################################################################
#               Outlier Analysis					       	   #
################################################################

#  #We are skipping outliers analysis becoz we already have a Class Imbalance issue.

################################################################
#               Feature Selection                              #
################################################################

#Here we will use corrgram to find corelation

##Correlation plot
#library('corrgram')

corrgram(train[,num_index],
         order = F,  #we don't want to reorder
         upper.panel=panel.pie,
         lower.panel=panel.shade,
         text.panel=panel.txt,
         main = 'CORRELATION PLOT')

#We can see that the highly corr related vars in plot are marked in dark blue. 
#Dark blue color means highly positive correlation

##------------------Chi Square Test--------------------------##

for(i in cat_col){
  print(names(cat_data[i]))
  print((chisq.test(table(cat_data$Churn,cat_data[,i])))[3])  #printing only pvalue
}

#-----------------Removing Highly Corelated and Independent var----------------------
train = subset(train,select= -c(state,total_day_charge,total_eve_charge,
                                total_night_charge,total_intl_charge))

test = subset(test,select= -c(state,total_day_charge,total_eve_charge,
                              total_night_charge,total_intl_charge))
							  
################################################################
#               Feature Scaling		                     			   #
################################################################

#all numeric var
num_index = sapply(train, is.numeric)
num_data = train[,num_index]
num_col = colnames(num_data) #storing all the column name


#Checking Data for Continuous Variables

################  Histogram   ##################
hist(train$total_day_calls)
hist(train$total_day_minutes)
hist(train$account_length)

#Most of the data is uniformly distributed
#Hence we shall be using data Standardization/Z-Score here

for(i in num_col){
  print(i)
  train[,i] = (train[,i] - mean(train[,i]))/sd(train[,i])
  test[,i] = (test[,i] - mean(test[,i]))/sd(test[,i])
}

################################################################
#          		        Sampling of Data        			           #
################################################################

# #Divide data into train and test using stratified sampling method

#install.packages('caret')
#library(caret)
set.seed(101)
split_index = createDataPartition(train$Churn, p = 0.66, list = FALSE)
trainset = train[split_index,]
validation_set  = train[-split_index,]

#Checking Train Set Target Class
table(trainset$Churn)
# 1    2 
# 1881  319 


# #Clearly our data has target class imbalance issue 
# Synthetic Over Sampling the minority class & Under Sampling Majority Class can be applied to have a good Training Set
# #library(ROSE)  #---> Lib for Over and Under Sampling

trainset <- ROSE(Churn~.,data = trainset,p = 0.5,seed = 101)$data     
table(trainset$Churn) 
# 1 = 1101  2 = 1099

############################################################################################################################################################
# #              Basic approach for ML - Models
# # We will first get a basic idea of how different models perform on our preprocesed data and then select the best model and make it more efficient for our Dataset
############################################################################################################################################################

# #Calculating FNR,FPR,Accuracy
calc <- function(cm){
  TN = cm[1,1]
  FP = cm[1,2]
  FN = cm[2,1]
  TP = cm[2,2]
  # #calculations
  print(paste0('Accuracy :- ',((TN+TP)/(TN+TP+FN+FP))*100))
  print(paste0('FNR :- ',((FN)/(TP+FN))*100))
  print(paste0('FPR :- ',((FP)/(TN+FP))*100))
  print(paste0('FPR :- ',((FP)/(TN+FP))*100))
  print(paste0('precision :-  ',((TP)/(TP+FP))*100)) 
  print(paste0('recall//TPR :-  ',((TP)/(TP+FP))*100))
  print(paste0('Sensitivity :-  ',((TP)/(TP+FN))*100))
  print(paste0('Specificity :-  ',((TN)/(TN+FP))*100))
  plot(cm)
}


### ##----------------------- Random Forest ----------------------- ## ###

#install.packages('randomForest')
#library('randomForest')
set.seed(101)
RF_model = randomForest(Churn ~ ., trainset,ntree= 500,importance=T,type='class')
plot(RF_model)
#Predict test data using random forest model
RF_Predictions = predict(RF_model, validation_set[,-15])

##Evaluate the performance of classification model
cm_RF = table(validation_set$Churn,RF_Predictions)
confusionMatrix(cm_RF)
calc(cm_RF)
plot(RF_model)

# Result of validaton-
# [1] "Accuracy :- 86.2312444836717"
# [1] "FNR :- 17.6829268292683"
# [1] "FPR :- 13.1062951496388"
# [1] "precision :-  51.5267175572519"
# [1] "recall//TPR :-  51.5267175572519"s
# [1] "Sensitivity :-  82.3170731707317"
# [1] "Specificity :-  86.8937048503612"

### ##----------------------- LOGISTIC REGRESSION ----------------------- ## ###
set.seed(101)
logit_model = glm(Churn ~., data = trainset, family =binomial(link="logit")) 
summary(logit_model)
#Prediction
logit_pred = predict(logit_model,newdata = validation_set[,-15],type = 'response')

#Converting Prob to number or class
logit_pred = ifelse(logit_pred > 0.5, 2,1)
#logit_pred = as.factor(logit_pred)
##Evaluate the performance of classification model
cm_logit = table(validation_set$Churn, logit_pred)
confusionMatrix(cm_logit)
calc(cm_logit)
plot(logit_model)
#roc(validation_set$Churn~logit_pred)

# Result on validaton set
# [1] "Accuracy :- 78.1994704324801"
# [1] "FNR :- 28.6585365853659"
# [1] "FPR :- 20.6398348813209"
# [1] "precision :-  36.9085173501577"
# [1] "recall//TPR :-  36.9085173501577"
# [1] "Sensitivity :-  71.3414634146341"
# [1] "Specificity :-  79.360165118679"
# ROC = 0.8017

### ##----------------------- KNN ----------------------- ## ###
######I have commented the code as I was facing some issue while final runnning of the file. Approach to be followed is the one used in below code######

#set.seed(101)
#library(class)

##Predicting Test data
#knn_Pred = knn(train = trainset[,1:14],test = validation_set[,1:14],cl = trainset$Churn, k = 5,prob = T)

#Confusion matrix
#cm_knn = table(validation_set$Churn,knn_Pred)
#confusionMatrix(cm_knn)
#calc(cm_knn)

# Result on validaton set
# [1] "Accuracy :- 78.8172992056487"
# [1] "FNR :- 46.3414634146341"
# [1] "FPR :- 16.9246646026832"
# [1] "precision :-  34.9206349206349"
# [1] "recall//TPR :-  34.9206349206349"
# [1] "Sensitivity :-  53.6585365853659"
# [1] "Specificity :-  83.0753353973168"



### ##----------------------- Naive Bayes ----------------------- ## ###

# library(e1071) #lib for Naive bayes
set.seed(101)
#Model Development and Training
naive_model = naiveBayes(Churn ~., data = trainset, type = 'class')
#prediction
naive_pred = predict(naive_model,validation_set[,1:14])

#Confusion matrix
cm_naive = table(validation_set[,15],naive_pred)
confusionMatrix(cm_naive)
calc(cm_naive)

# Result on validaton set
# [1] "Accuracy :- 83.1421006178288"
# [1] "FNR :- 29.2682926829268"
# [1] "FPR :- 14.7574819401445"
# [1] "precision :-  44.7876447876448"
# [1] "recall//TPR :-  44.7876447876448"
# [1] "Sensitivity :-  70.7317073170732"
# [1] "Specificity :-  85.2425180598555"


###########################################################################################
#               Final Random Forest Model with tuning parameters						  #
###########################################################################################

set.seed(101)
final_model = randomForest(Churn~.,data = trainset,ntree=800,mtry=4,importance=TRUE,type = 'class')
final_validation_pred = predict(final_model,validation_set[,-15])
cm_final_valid = table(validation_set[,15],final_validation_pred)
confusionMatrix(cm_final_valid)
calc(cm_final_valid)
#Result on validation set after parameter tuning
# [1] "Accuracy :- 86.4960282436011"
# [1] "FNR :- 17.0731707317073"
# [1] "FPR :- 12.8998968008256"
# [1] "FPR :- 12.8998968008256"
# [1] "precision :-  52.1072796934866"
# [1] "recall//TPR :-  52.1072796934866"
# [1] "Sensitivity :-  82.9268292682927"
# [1] "Specificity :-  87.1001031991744"


#Variable Importance
importance(final_model) #builtin function in Random forest lib
varImpPlot(final_model) #builtin func

#Plotting ROC curve and Calculate AUC metric
# library(pROC)
PredictionwithProb <-predict(final_model,validation_set[,-15],type = 'prob')
auc <- auc(validation_set$Churn,PredictionwithProb[,2])
auc
# # AUC = 89.47
plot(roc(validation_set$Churn,PredictionwithProb[,2]))

###################################################################################
#        Final Prediction On test Data set 										                    #
###################################################################################

#rmExcept(c("final_model","train","test","train_Original","test_Original","calc"))

set.seed(101)
final_test_pred = predict(final_model,test[,-15])
cm_final_test = table(test[,15],final_test_pred)
confusionMatrix(cm_final_test)
calc(cm_final_test)

# #Final Test Prediction
# [1] "Accuracy :- 85.9028194361128"
# [1] "FNR :- 15.625"
# [1] "FPR :- 13.8600138600139"
# [1] "precision :-  48.586118251928"
# [1] "recall//TPR :-  48.586118251928"
# [1] "Sensitivity :-  84.375"
# [1] "Specificity :-  86.1399861399861"

#Plotting ROC curve and Calculate AUC metric
# library(pROC)
finalPredictionwithProb <-predict(final_model,test[,-15],type = 'prob')
auc <- auc(test$Churn,finalPredictionwithProb[,2])
auc
# # AUC = 91.74
plot(roc(test$Churn,finalPredictionwithProb[,2]))

########################################################################################
#             			 Saving output to file										   #
########################################################################################

test_Original$predicted_output <- final_test_pred
test_Original$predicted_output <- gsub(1,"False",test_Original$predicted_output)
test_Original$predicted_output <- gsub(2,"True",test_Original$predicted_output)

#Entire Comparison
write.csv(test_Original,'C:/Users/Click/Desktop/project1_custchurn/Output_R.csv',row.names = F)

#Phonenumber and Churning class and probab
submit <- data.frame(test_Original$state,
                     test_Original$area.code,
                     test_Original$international.plan,
                     test_Original$voice.mail.plan,
                     test_Original$phone.number,
                     test_Original$predicted_output,
                     finalPredictionwithProb[,1],
                     finalPredictionwithProb[,2])

colnames(submit) <- c("State","Area Code","International Plan","Voice Mail Plan","Phone_Number",
                      "Predicted_Output","Probability_of_False","Probability_of_True")

write.csv(submit,file = 'C:/Users/Click/Desktop/project1_custchurn/FinalChurn_R.csv',row.names = F)
rm(list = ls())

