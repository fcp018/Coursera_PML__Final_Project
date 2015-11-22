# Coursera PML Final Project Report
Cp  

<br>

### Summary  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


<br> 

### Load necessary packages  
In the intial step, we load the necessary packages and download and read the data.


```
## Loading required package: lattice
## Loading required package: ggplot2
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```
<br>

### Downloading and load data into memory


```r
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training_raw <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing_raw <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```

### Next, we look at dimension of the raw datasets and first 10 rows.


```r
dim(training_raw)
```

```
## [1] 19622   160
```

```r
dim(testing_raw)
```

```
## [1]  20 160
```
As shown above, there are 19622 observations for 160 variables in the training data set and 20 observations for 160 variable in the testing data set.  The variable to predict is the "classe" variable in the training set. Viewing the first few rows of the raw data sets with the "head" function shows that there are many columns with missing values and columns that are not necessary for prediction.  We will clean these in next step.

<br>

### Cleaning/processing the data
In the following steps, we remove missing values (columns containing "NA") and unnecessary variables(like the first 7 columns).


```r
cleanTrain<-training_raw[,-seq(1:7)]
cleanTest<-testing_raw[,-seq(1:7)]
hasNA<-as.vector(sapply(cleanTrain[,1:152],function(x) {length(which(is.na(x)))!=0}))
cleanTrain<-cleanTrain[,!hasNA]
cleanTest<-cleanTest[,!hasNA]

dim(cleanTrain)
```

```
## [1] 19622    53
```

```r
dim(cleanTest)
```

```
## [1] 20 53
```
The cleaning steps reduced the training data to 19622 observations with 53 variables(including "classe") and the testing data set to 20 observations with 53 variables. 

<br>

### Creating Data Partitions
Next we divide the cleaned data set into training set(70%) and a validation set (30%). The validation data set will be used to conduct cross validation.  


```r
inTrain <- createDataPartition(cleanTrain$classe, p=0.70, list=F)
Train_data <- cleanTrain[inTrain, ]
Test_data <- cleanTrain[-inTrain, ]
```

<br>

### Prediction Modeling
We will compare two different prediction models, **Decision Tree** and **Random Forest**, and pick the one the method that produces the most accurate model.  We expect that Random Forest algorithm will be most accurate since it is robust to correlated covariates and outliers and it automatically selects important variables. 

<br>

### First, Decision Tree model  


```r
modDTfit <- rpart(classe ~ ., data=Train_data, method="class")
```

<br>

### Now we estimate model preformance using the validation set


```r
modDTpredict <- predict(modDTfit, Test_data, type = "class")
confusionMatrix(modDTpredict, Test_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1537  172   20   45   17
##          B   42  638   39   60   81
##          C   52  156  832  153  124
##          D   16   83   77  625   62
##          E   27   90   58   81  798
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7528          
##                  95% CI : (0.7415, 0.7637)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6867          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9182   0.5601   0.8109   0.6483   0.7375
## Specificity            0.9397   0.9532   0.9002   0.9516   0.9467
## Pos Pred Value         0.8582   0.7419   0.6317   0.7242   0.7571
## Neg Pred Value         0.9665   0.9003   0.9575   0.9325   0.9412
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2612   0.1084   0.1414   0.1062   0.1356
## Detection Prevalence   0.3043   0.1461   0.2238   0.1466   0.1791
## Balanced Accuracy      0.9289   0.7567   0.8556   0.8000   0.8421
```
At 75%, the accuracy of this model is not very good.

<br>

### Next, we create a model with Random Forest algorithm


```r
modRFfit <- randomForest(classe ~. , data=Train_data)
modRFfit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = Train_data) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.49%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    4    0    0    1 0.001280082
## B   13 2640    5    0    0 0.006772009
## C    0   12 2381    3    0 0.006260434
## D    1    0   19 2231    1 0.009325044
## E    0    0    2    6 2517 0.003168317
```


<br>

### Estimating the random forest model performance with validation data set

```r
modRFpredict <- predict(modRFfit, Test_data, type = "class")

confusionMatrix(modRFpredict, Test_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    7    0    0    0
##          B    3 1132    7    0    0
##          C    0    0 1015   11    1
##          D    0    0    4  953    6
##          E    0    0    0    0 1075
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.991, 0.9953)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9939   0.9893   0.9886   0.9935
## Specificity            0.9983   0.9979   0.9975   0.9980   1.0000
## Pos Pred Value         0.9958   0.9912   0.9883   0.9896   1.0000
## Neg Pred Value         0.9993   0.9985   0.9977   0.9978   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1924   0.1725   0.1619   0.1827
## Detection Prevalence   0.2851   0.1941   0.1745   0.1636   0.1827
## Balanced Accuracy      0.9983   0.9959   0.9934   0.9933   0.9968
```
As expected, we see more accurate resutls with Random Forest model.  

<br><br>

### Calculating accuary and out of sample error


```r
accuracy <- postResample(modRFpredict, Test_data$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9933730 0.9916169
```

```r
oose <- 1 - as.numeric(confusionMatrix(Test_data$classe, modRFpredict)$overall[1])
oose
```

```
## [1] 0.006627018
```
We see excellent results using validation set with accuracy of  99.3 %  and the estimated out-of-sample error is  0.7 % .  We will use the Random Forest algorithm model to predict on the test data set.

<br><br>

## Final Prediction for test set
In this step, we apply the prediction model to the orginal testing data set without the "problem_id" variable.


```r
Finaltest <- predict(modRFfit, cleanTest[, -length(names(cleanTest))])
Finaltest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
<br>

Function to generate files with predictions to submit for assignment

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(Finaltest)
```

