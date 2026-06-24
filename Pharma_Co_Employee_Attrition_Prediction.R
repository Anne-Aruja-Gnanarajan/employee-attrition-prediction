# Pharma Co Prediction of Employee Attribution
# by Anne Gnanarajan
suppressMessages(library(pROC))
library(rpart)
library(rpart.plot)

# Reading employee data 

employees <- read.csv(
  file = '/course/data/employees2.csv',
  stringsAsFactors = TRUE
)

head(employees)
str(employees)

# Changing attrition to 0/1 (for modelling purposes) rather than "Yes"/"No"
employees$Attrition <- ifelse(employees$Attrition == "Yes", 1, 0)

str(employees)

# Create the ten folds
##############################################################
set.seed(57)
employees$Fold <- sample(rep(1:10, length.out=nrow(employees)))

# Define a function to do the cross-validation
# LOGISTIC REGRESSION MODEL
###########################################################################
cv_score <- function(predictors) {
  scores <- c()
  for (fold in 1:10) {
    training = employees[employees$Fold != fold,]
    validation = employees[employees$Fold == fold,]
    model <- glm(
      # Build the formula from the predictors
      formula = paste("Attrition ~ ", paste(predictors, collapse = " + "), sep = ""),
      data = training,
      family = "binomial"
    )
    validation$Probability <- predict(
      object = model,
      newdata = validation,
      type = "response"
    )
    auc <- roc(validation$Attrition ~ validation$Probability, quiet = TRUE)$auc
    scores <- c(scores, auc)
    #message(paste("Predictors ",predictors, "Fold ", fold, sep = ""))
  }
  return(mean(scores))
}

# Define a function to do the stepping forward
###########################################################################
step_forward <- function(current_predictors, possible_predictors) {
  
  # If there are no possible predictors, return the current_predictors
  if (length(possible_predictors) == 0) return(current_predictors)
  
  # Get the cross-validation score of the current predictors
  if (length(current_predictors > 0)) current_score <- cv_score(current_predictors)
  else current_score = 0
  
  # Show the current predictors and their cross-validation score
  message("Current predictors: ", paste(current_predictors, collapse=" + "))
  message("Current score: ", current_score)
  
  # Create a vector in which to put the new scores
  new_scores <- c()
  
  # Try adding each of the possible predictors
  for (predictor_to_try in possible_predictors) {
    # Add the predictor to the current predictors
    new_predictors <- c(current_predictors, predictor_to_try)
    # Find the cross-validation score of the new predictors
    new_score <- cv_score(new_predictors)
    # Show the score
    message("With ", predictor_to_try, ": ", new_score)
    # Add it to the vector of new scores
    new_scores[predictor_to_try] <- new_score
  }
  
  # If the best new score is better than the current score
  if (max(new_scores) > (current_score + 0.01)) {
    # Find the predictor to add
    predictor_to_add <- names(new_scores)[which.max(new_scores)]
    # Output an explanatory message
    message("Adding ", predictor_to_add, "\n")
    # Add it to the current predictors
    current_predictors <- c(current_predictors, predictor_to_add)
    # Remove it from the possible predictors
    possible_predictors <- possible_predictors[possible_predictors != predictor_to_add]
    # Step forward again
    return(step_forward(current_predictors, possible_predictors))
  }
  
  # Otherwise
  else {
    message("Nothing to add\n")
    return(current_predictors)
  }
}

# Define the starting predictors and the possible predictors
##########################################################################
starting_predictors <- c()
possible_predictors <- c(
    'Age',
    'BusinessTravel', #### 5
    'Department',
    'DistanceFromHome',
    'Education',
    'EnvironmentSatisfaction',
    'JobInvolvement', #### 4
    'JobRole',
    'MaritalStatus', #### 3
    'MonthlyIncome', #### 1
    'NumCompaniesWorked',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StockOptionLevel',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'WorkLifeBalance',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    'Gender',
    'OverTime' #### 2
    )
print(possible_predictors)

# Use step_forward to find the best predictors
best_predictors <- step_forward(starting_predictors, possible_predictors)

# Show the results
message("Best predictors: ", paste(best_predictors, collapse = " + "))
message("Cross-validation score: ", cv_score(best_predictors))


####### Training the model with the final selection of variables, across the full dataset
##############################################################################
model <- glm(
    formula = Attrition ~ MonthlyIncome + OverTime + MaritalStatus + JobInvolvement, #best predictors
    data = employees,
    family = "binomial"
)
model
summary(model)

#final predictions
employees$Probability <- predict(
    object = model,
    newdata = employees,
    type = "response"
)
head(employees)

results <- roc(
  employees$Attrition ~ employees$Probability, 
  plot = TRUE,
  print.thres = "best",
  main = "ROC Curve for Logistic Regression Model Trained on Full Dataset")
results

#Create predictions based on best threshold
employees$Prediction <- ifelse(employees$Probability>= 0.196, 1, 0) # best threshold
#Create confusion matrix
conf_matrix <- table(employees$Prediction, employees$Attrition)
conf_matrix
# Calculate the accuracy
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
# Show the accuracy
message("Accuracy: ", accuracy)

### Decision Tree Model
##################################################################

# Define a function to do the cross-validation
###########################################################################
cv_score <- function(predictors) {
  scores <- c()
  for (fold in 1:10) {
    training = employees[employees$Fold != fold,]
    validation = employees[employees$Fold == fold,]
    model <- rpart(
      # Build the formula from the predictors
      formula = paste("Attrition ~ ", paste(predictors, collapse = " + "), sep = ""),
      data = training,
      method = "class",
      maxdepth = 5
    )
    validation$Probability <- predict(
      object = model,
      newdata = validation,
      type = "prob"
    )
    auc <- roc(validation$Attrition ~ validation$Probability[,2], quiet = TRUE)$auc
    scores <- c(scores, auc)
    #message(paste("Predictors ",predictors, "Fold ", fold, sep = ""))
  }
  return(mean(scores))
}

# Use step_forward to find the best predictors
best_predictors <- step_forward(starting_predictors, possible_predictors)

# Show the results
message("Best predictors: ", paste(best_predictors, collapse = " + "))
message("Cross-validation score: ", cv_score(best_predictors))

####### Training the model with the final selection of variables, across the full dataset
##############################################################################
model <- rpart(
  formula = Attrition ~ MonthlyIncome + OverTime + TotalWorkingYears + EnvironmentSatisfaction, # best predictors
  data = employees,
  method = "class",
  maxdepth = 5
)
model
summary(model)

# final predictions
employees$Probability <- predict(
  object = model,
  newdata = employees,
  type = "prob"
)
head(employees)

results <- roc(
  employees$Attrition ~ employees$Probability[,2], 
  plot = TRUE,
  print.thres = "best",
  main = "ROC Curve for Decision Tree Model Trained on Full Dataset")
results

#Plotting decision tree
rpart.plot(
  model, 
  type = 4
)

#Create predictions based on best threshold
employees$Prediction <- ifelse(employees$Probability[,2]>= 0.196, 1, 0) # best threshold
#Create confusion matrix
conf_matrix <- table(employees$Prediction, employees$Attrition)
conf_matrix
# Calculate the accuracy
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
# Show the accuracy
message("Accuracy: ", accuracy)


###### Plotting ##################################
###################################################

## Plotting Monthly Income and Attrition
# Creating employee income brackets
employees$IncomeBracket[employees$MonthlyIncome<=5000] <- 5000
employees$IncomeBracket[employees$MonthlyIncome>5000 & employees$MonthlyIncome<=10000] <- 10000
employees$IncomeBracket[employees$MonthlyIncome>10000 & employees$MonthlyIncome<=15000]<- 15000
employees$IncomeBracket[employees$MonthlyIncome>15000 & employees$MonthlyIncome<=20000]<- 20000
employees$IncomeBracket[employees$MonthlyIncome>20000 & employees$MonthlyIncome<=25000]<- 25000

head(employees)
# Summarising employees by Income and Attrition
income_attrition <- aggregate(
  JobRole ~ IncomeBracket + Attrition,
  data = employees,
  FUN = length
)
income_attrition

#combining table for males with table for females for plotting
income_attrition1 <- merge(
  x = income_attrition[income_attrition$Attrition==0,], 
  y = income_attrition[income_attrition$Attrition==1,], 
  by = "IncomeBracket", 
  all = TRUE)
income_attrition1

# Calculating percentages
income_attrition1$Retention <- (income_attrition1$JobRole.x/(income_attrition1$JobRole.x+income_attrition1$JobRole.y))*100
income_attrition1$Attrition <- (income_attrition1$JobRole.y/(income_attrition1$JobRole.x+income_attrition1$JobRole.y))*100

income_attrition1

# creating bar plot
barplot(
  cbind(Attrition, Retention) ~ IncomeBracket,
  data = income_attrition1,
  col = c('pink', 'lightskyblue2'),
  ylab = "Percent of Employees",
  xlab = "Income Bucket (less than or equal to)",
  #las = 3,
  main = "Employee Attrition by Monthly Income Bucket",
  cex.main = 1.2, cex.lab = 1.2, cex.axis = 1.1
)

# Adding legend for colours
legend(
    # Put the legend in the top middle
    "top",
    legend = c("Attrition", "Retention"),
    col = c("pink", "lightskyblue2"),
    pch = 19, 
    pt.cex = 2,
    bg = "white"
)

# Summarising employees by Overtime and Attrition
overtime_attrition <- aggregate(
  JobRole ~ OverTime + Attrition,
  data = employees,
  FUN = length
)
overtime_attrition

#combining table for males with table for females for plotting
overtime_attrition1 <- merge(
  x = overtime_attrition[overtime_attrition$Attrition==0,], 
  y = overtime_attrition[overtime_attrition$Attrition==1,], 
  by = "OverTime", 
  all = TRUE)
overtime_attrition1

# Calculating percentages
overtime_attrition1$Retention <- (overtime_attrition1$JobRole.x/(overtime_attrition1$JobRole.x+overtime_attrition1$JobRole.y))*100
overtime_attrition1$Attrition <- (overtime_attrition1$JobRole.y/(overtime_attrition1$JobRole.x+overtime_attrition1$JobRole.y))*100

overtime_attrition1

# creating bar plot
barplot(
  cbind(Attrition, Retention) ~ OverTime,
  data = overtime_attrition1,
  col = c('pink', 'lightskyblue2'),
  ylab = "Percent of Employees",
  xlab = "Overtime",
  #las = 3,
  main = "Employee Attrition by Overtime",
  cex.main = 1.2, cex.lab = 1.2, cex.axis = 1.1
)

legend(
    # Put the legend in the top middle
    "top",
    legend = c("Attrition", "Retention"),
    col = c("pink", "lightskyblue2"),
    pch = 19, 
    pt.cex = 2,
    bg = "white"
)
