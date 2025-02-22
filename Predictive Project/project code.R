# Load necessary libraries
#final for project predictive analysis using Decision Tree,Knn and Logistic Regression.library(class)
library(caret)
library(e1071)
library(rpart)
library(ggplot2)  # Added for visualization

# Load the data
wbcd <- read.csv("C:/SEMESTER-5/INT234/wisc_bc_data (1).csv", stringsAsFactors = FALSE)

# Remove the ID column
wbcd <- wbcd[-1]
table(wbcd$diagnosis)

# Convert diagnosis to a factor
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

# Normalize the features
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))

# Split into train and test sets
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

# KNN Model
wbcd_test_pred_knn <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 21)
knn_metrics <- confusionMatrix(wbcd_test_pred_knn, wbcd_test_labels)
print(knn_metrics)

# Logistic Regression Model
log_model <- glm(diagnosis ~ ., data = wbcd[1:469, ], family = "binomial")
log_pred <- factor(ifelse(predict(log_model, wbcd[470:569, ], type = "response") > 0.5, "Malignant", "Benign"), levels = c("Benign", "Malignant"))
log_metrics <- confusionMatrix(log_pred, wbcd_test_labels)
print(log_metrics)

# Decision Tree Model
dt_model <- rpart(diagnosis ~ ., data = wbcd[1:469, ], method = "class")
dt_pred <- predict(dt_model, wbcd[470:569, ], type = "class")
dt_metrics <- confusionMatrix(dt_pred, wbcd_test_labels)
print(dt_metrics)

# Function to calculate performance metrics (accuracy, precision, recall, F1 score, error rate)
calculate_metrics <- function(true_labels, predicted_labels) {
  cm <- confusionMatrix(predicted_labels, true_labels)
  
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1_score <- 2 * ((precision * recall) / (precision + recall))
  error_rate <- 1 - accuracy
  
  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    error_rate = error_rate
  ))
}

# Compare models (KNN, Logistic Regression, and Decision Tree)
knn_metrics_values <- calculate_metrics(wbcd_test_labels, wbcd_test_pred_knn)
log_metrics_values <- calculate_metrics(wbcd_test_labels, log_pred)
dt_metrics_values <- calculate_metrics(wbcd_test_labels, dt_pred)

# Print comparison
cat("KNN Metrics:\n")
print(knn_metrics_values)
cat("\nLogistic Regression Metrics:\n")
print(log_metrics_values)
cat("\nDecision Tree Metrics:\n")
print(dt_metrics_values)

# Visualization: Model Metric Comparison
metrics_comparison_plot <- function(metrics_list, metric_name) {
  metrics_df <- data.frame(
    Model = c("KNN", "Logistic Regression", "Decision Tree"),
    Metric = c(metrics_list[[1]][[metric_name]], metrics_list[[2]][[metric_name]], metrics_list[[3]][[metric_name]])
  )
  
  ggplot(metrics_df, aes(x = Model, y = Metric, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    ggtitle(paste("Model", metric_name, "Comparison")) +
    ylab(metric_name) +
    theme_minimal()
}

# Plot for Accuracy Comparison
metrics_comparison_plot(list(knn_metrics_values, log_metrics_values, dt_metrics_values), "accuracy")

# Plot for Precision Comparison
metrics_comparison_plot(list(knn_metrics_values, log_metrics_values, dt_metrics_values), "precision")

# Plot for Recall Comparison
metrics_comparison_plot(list(knn_metrics_values, log_metrics_values, dt_metrics_values), "recall")

# Plot for F1 Score Comparison
metrics_comparison_plot(list(knn_metrics_values, log_metrics_values, dt_metrics_values), "f1_score")

# Plot for Error Rate Comparison
metrics_comparison_plot(list(knn_metrics_values, log_metrics_values, dt_metrics_values), "error_rate")
