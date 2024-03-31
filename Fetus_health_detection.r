library(dplyr)
library(tidyr)
library(caret)


# I.Loading the dataset
your_data <- read.csv("C:\\Users\\Rohit\\Desktop\\IDA Project-2\\IDA Project-1\\fetal_health.csv")

####################### II. Exploratory Data Analysis  #######################
column_names <- names(your_data)
print(column_names)
head(your_data)


par(mfrow = c(5, 5), mar = c(2, 1, 1, 1))  

for (col in colnames(your_data)) {
  hist(your_data[[col]], main = col, xlab = "", col = "lightblue", border = "black")
}

summary(your_data)


############################# III.Data Cleaning ############################

# Handle Missing Values
your_data <- na.omit(your_data)
colSums(is.na(your_data))

# Check for missing values in the entire dataset
if(any(is.na(your_data)) == FALSE){
  print('No missing values')
}

# Assuming 'your_data' is your data frame
num_duplicates <- sum(duplicated(your_data))
cat("Number of Duplicate Rows: ", num_duplicates, "\n")

# Display duplicated rows
duplicated_rows <- your_data[duplicated(your_data), ]
print(duplicated_rows)


# Display all occurrences of duplicated rows
all_duplicates <- your_data[duplicated(your_data) | duplicated(your_data, fromLast = TRUE), ]
print(all_duplicates)

# Assuming 'your_data' is your data frame
your_data <- unique(your_data)

############################### Feature Scaling ############################

# Remove duplicates and create a new data frame
unique_data <- unique(your_data)
print(unique_data)

column_names <-colnames(unique_data)

threshold <- 1.5

lower_bounds <- c() 
upper_bounds <- c() 

for (column in column_names) { 
  column_iqr <- IQR(your_data[[column]]) 
  column_lower_bound <- quantile(your_data[[column]], 0.25) - threshold * column_iqr 
  column_upper_bound <- quantile(your_data[[column]], 0.75) + threshold * column_iqr 
  
  lower_bounds <- c(lower_bounds, column_lower_bound) 
  upper_bounds <- c(upper_bounds, column_upper_bound) 
}

for (i in 1:length(column_names)) { 
  column <- column_names[i] 
  lower_bound <- lower_bounds[i] 
  upper_bound <- upper_bounds[i] 
  
  your_data <- your_data[your_data[[column]] >= lower_bound & your_data[[column]] <= upper_bound, ] 
} 

#levels(unique_data$prolongued_decelerations)
boxplot(unique_data)

boxplot_data <- your_data[, column_names, drop = FALSE] 
boxplot(boxplot_data, names = column_names, main = "Box Plot of Numerical Columns")

correlation_matrix <- cor(unique_data)
print(correlation_matrix)
heatmap(correlation_matrix, 
        col = colorRampPalette(c("blue", "white", "red"))(100), 
        main = "Correlation Heatmap", 
        xlab = "Variables", 
        ylab = "Variables"
)

threshold <- 0.8

library(caret)
# Find highly correlated features
highly_correlated <- findCorrelation(correlation_matrix, cutoff = threshold)
highly_correlated

# Remove highly correlated features
reduced_data <- unique_data[, -highly_correlated]



# Set the threshold for variance (adjust as needed)
threshold <- 0.05

# Identify low-variance features
low_variance_features <- nearZeroVar(your_data, saveMetrics = TRUE, freqCut = threshold)

# Extract names of low-variance features
low_variance_feature_names <- rownames(low_variance_features)

# Remove low-variance features from the dataset
your_data_filtered <- your_data[, -which(names(your_data) %in% low_variance_feature_names)]



######################### Data Transformation ##############################

library(e1071)
categorical_columns <- sapply(your_data, function(x) is.factor(x) || is.character(x))

# Function to decide whether to standardize or normalize
decide_scaling <- function(data) {
  # Calculate the range of each feature
  feature_ranges <- apply(data, 2, function(x) diff(range(x)))
  
  # Calculate the mean and standard deviation of each feature
  feature_means <- apply(data, 2, mean)
  feature_sds <- apply(data, 2, sd)
  
  # Calculate a metric based on the range, mean, and standard deviation
  scaling_metric <- feature_ranges / (feature_means * feature_sds)
  
  # Decide based on the scaling metric
  if (all(scaling_metric > 0.1)) {
    return("Standardize")
  } else {
    return("Normalize")
  }
}

# Decide whether to standardize or normalize
scaling_decision <- decide_scaling(your_data)

# Display the decision
print(paste("Based on the metric, you should:", scaling_decision))

# Assuming 'your_data' is your data frame with numeric columns to be normalized
numeric_columns <- sapply(your_data, is.numeric)

# Extract only the numeric columns for normalization
your_data_numeric <- your_data[, numeric_columns]

# Normalize the numeric columns using the scale() function
normalized_data <- scale(your_data_numeric)

# Combine the normalized numeric columns with non-numeric columns
your_data_normalized <- cbind(your_data[, !numeric_columns], normalized_data)

# Display the original and normalized datasets
print("Original Data:")
print(head(your_data))
print("\nNormalized Data:")
print(head(your_data_normalized))



  
  
  code:
  library(caTools)
library(caret)
library(e1071)
library(gpairs)
library(ggplot2)
library(ModelMetrics)
library(ROCR)

#  Mean Function
mean <- function(data){
  sum <- 0
  for(i in 1:length(data)){
    sum <- sum + data[i]
  }
  return(sum/length(data))
}

#  Standard Deviation
std <- function(data,mean){
  #sum <- 0
  #for(i in 1:length(data)){
  #  sum <- sum + (data[i]-mean)^2
  #}
  #return(sqrt(sum/length(data)))
  return(sd(data, na.rm=TRUE))
}

# Normalization
norm <- function(x, meanVal, stdVal){
  return (exp(-(x-meanVal)^2/(2*stdVal^2))/(stdVal*sqrt(2*pi)))
}



#naive bayes Classifier
predict <- function(train_data,test_x,continous_cols){
  
  train_data.y0 <- subset(train_data,subset=train_data$fetal_health==1)
  train_data.y1 <- subset(train_data,subset=train_data$fetal_health==2)
  train_data.y2 <- subset(train_data,subset=train_data$fetal_health==3)
  
  continous_lkh <- list()
  for(i in continous_cols){
    continous_lkh[[i]] <- list()
    mean <- mean(train_data.y0[[i]])
    std <- sd(train_data.y0[[i]],na.rm=TRUE)
    continous_lkh[[i]][["0"]] <- list("mean" = mean,"std" = std)
    mean <- mean(train_data.y1[[i]])
    std <- sd(train_data.y1[[i]],na.rm=TRUE)
    continous_lkh[[i]][["1"]] <- list("mean" = mean,"std" = std)
    mean <- mean(train_data.y2[[i]])
    std <- sd(train_data.y2[[i]],na.rm=TRUE)
    continous_lkh[[i]][["2"]] <- list("mean" = mean,"std" = std)
  }
  
  pred_y = c()
  prior_target = c(length(which(train_data$fetal_health==1))/nrow(train_data),length(which(train_data$fetal_health==2))/nrow(train_data),length(which(train_data$fetal_health==3))/nrow(train_data))
  for(row in 1:nrow(test_x)){
    prob.0 <- prior_target[1] # Posterior Probability P(target=1/x)
    prob.1 <- prior_target[2]
    prob.2 <- prior_target[3]
    fetal_health <- 0
    for(i in continous_cols){
      test_val <- test_x[row,i]
      prob.0 <- prob.0 * norm(test_val,continous_lkh[[i]][["0"]][["mean"]],continous_lkh[[i]][["0"]][["std"]])
      prob.1 <- prob.1 * norm(test_val,continous_lkh[[i]][["1"]][["mean"]],continous_lkh[[i]][["1"]][["std"]])
      prob.2 <- prob.2 * norm(test_val,continous_lkh[[i]][["2"]][["mean"]],continous_lkh[[i]][["2"]][["std"]])
    }
    
    if (prob.0 > prob.1 & prob.0 > prob.2) {
      fetal_health <- 1
    } else if (prob.1 > prob.2) {
      fetal_health <- 2
    } else {
      fetal_health <- 3
    }
    
    pred_y[row] <- fetal_health
  }
  return(pred_y)
}



#k-fold cross validation
KFoldCrossVal <- function(dataset,k) {
  sprintf("Cross Validation with K = %d",k)
  len = nrow(dataset)
  accuracy = c()
  for(i in 1:k){
    from <- 1+as.integer(len/k)*(i-1)
    to <- as.integer(len/k)*i
    test_data <- dataset[from:to,]
    train_data <- rbind(dataset[0:(from-1),],dataset[(to+1):len,])
    
    test_x <- subset(test_data,select = -fetal_health)
    test_y <- subset(test_data,select = fetal_health)
    
    result <- data.frame(test_y)
    result$predicted_target <- predict(train_data,test_x,continous_cols)
    
    accuracy[i] <- length(which(result$fetal_health==result$predicted_target))/length(result$fetal_health)*100
  }
  return(mean(accuracy))
}


#importing dataset
dataset = read.csv("C:\\Users\\Rohit\\Desktop\\IDA Project-2\\IDA Project-1\\fetal_health.csv")
dataset = na.omit(dataset)
str(dataset)
summary(dataset)

continous_cols = c("baseline.value",                                       
                   "accelerations",                                         
                   "fetal_movement",                                        
                   "uterine_contractions",                                  
                   "light_decelerations",                                   
                   "prolongued_decelerations",                              
                   "abnormal_short_term_variability",                       
                   "mean_value_of_short_term_variability",                  
                   "percentage_of_time_with_abnormal_long_term_variability",
                   "mean_value_of_long_term_variability",                 
                   "histogram_width",                                       
                   "histogram_min",                                         
                   "histogram_max",                                         
                   "histogram_number_of_peaks",                            
                   "histogram_number_of_zeroes",                           
                   "histogram_mode",                                        
                   "histogram_mean",                                        
                   "histogram_median",                                      
                   "histogram_variance",                                    
                   "histogram_tendency"                                  
)

#k fold cross validation
avgAcc = c()
least.k = 3
max.k= 15
maxacc = 0
maxk=0
for(k in least.k:max.k){
  accuracy <- KFoldCrossVal(dataset,k)
  if(accuracy>maxacc){
    maxacc=accuracy
    maxk=k
  }
  avgAcc<- append(avgAcc,accuracy)
}

print(paste0("Highest Accuracy: ", maxacc, "for k= ", maxk))

print(avgAcc)
bestKvalue = which.max(avgAcc)+least.k-1
print(bestKvalue)

sprintf("So the acuracy with best k = %d value for k fold cross validation is %f",bestKvalue,avgAcc[[bestKvalue-least.k+1]])

split <- createDataPartition(dataset$fetal_health,p=0.75,list=FALSE)
td.size <- 0.75*nrow(dataset)
train_data <- dataset[1:td.size,]
test_data <- dataset[td.size:nrow(dataset),]
test_x <- subset(test_data, select = -fetal_health)
test_y <- subset(test_data, select = fetal_health)

result <- data.frame(test_y)
result$predicted_target <- predict(train_data,test_x,continous_cols)
write.csv(result,"C:\\Users\\mahan\\Downloads\\IDA Project\\result.csv", row.names = FALSE)

# Accuracy 
accuracy <- length(which(result$fetal_health==result$predicted_target))/length(result$fetal_health)*100
sprintf("accuracy: %f",accuracy)

# Confusion Matrix
confusion_matrix <- table(result$fetal_health,result$predicted_target,dnn=names(result))
print(confusion_matrix)

# Precision
precision <- ppv(result$fetal_health,result$predicted_target)
sprintf("Precision: %f",precision)

# Recall
recall <- recall(result$fetal_health,result$predicted_target)
sprintf("recall: %f",recall)

# F1 Score 
f1.score <- f1Score(result$fetal_health,result$predicted_target)
sprintf("F1 score: %f",f1.score)

