
install.packages("caret")
library(caret)


data <- read.csv("F:/exams.csv", header = TRUE, sep = ",")
print(data)

is.na(data)
sum(is.na(data))
colSums(is.na(data))

formula <- as.formula("gender ~ .")


dummy_data <- dummyVars(formula, data = data)
encoded_data <- as.data.frame(predict(dummy_data, newdata = data))


data <- cbind(data[, setdiff(names(data), "gender")], encoded_data)

numerical_cols <- c("math.score", "reading.score", "writing.score")
normalized_data <- data

normalized_data[, numerical_cols] <- scale(data[, numerical_cols])


print(normalized_data)


numerical_features <- c("math.score", "reading.score", "writing.score")
correlation_matrix <- cor(data[, numerical_features], use = "pairwise.complete.obs")

target_correlation <- cor(data[, numerical_features], data$math.score, use = "pairwise.complete.obs")
print(target_correlation)

correlation_threshold <- 0.1
print(correlation_threshold)

low_correlation_features <- numerical_features[abs(target_correlation) < correlation_threshold]
print(low_correlation_features)

data <- data[, !(names(data) %in% low_correlation_features)]
print(data)

categorize_performance <- function(score) {
  ifelse(score >= 60, "pass", "fail")
}
print(categorize_performance)

data$math_performance <- categorize_performance(data$math.score)
data$reading_performance <- categorize_performance(data$reading.score)
data$writing_performance <- categorize_performance(data$writing.score)

print(data)
print(data$math_performance)

set.seed(123)
train_ratio <- 0.8

train_indices <- createDataPartition(data$math.score, p = train_ratio, list = FALSE)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

k_values <- 1:10
ctrl <- trainControl(method = "cv", number = 10)

grid <- expand.grid(k = k_values)
train_model <- train(math_performance ~ ., data = train_data, method = "knn",
                     trControl = ctrl, tuneGrid = grid)

test_model <- train(math_performance ~ ., data = test_data, method = "knn",
                    trControl = ctrl, tuneGrid = grid)
print(train_model)
print(test_model)

train_k <- train_model$bestTune$k

train_data$math_performance <- categorize_performance(train_data$math.score)

predictors_for_train <- setdiff(names(train_data), c("math_performance", "math.score"))

final_train_model <- train(math_performance ~ ., data = train_data[, c("math_performance", predictors_for_train)],
                           method = "knn", trControl = ctrl, tuneGrid = data.frame(k = train_k))
print(train_k)

test_k <- test_model$bestTune$k
predictors_for_test <- setdiff(names(test_data), c("math_performance", "math.score"))
final_test_model <- train(math_performance ~ ., data = test_data[, c("math_performance", predictors_for_test)],
                          method = "knn", trControl = ctrl, tuneGrid = data.frame(k = test_k))
print(test_k)

prediction_traindata <- predict(final_train_model, newdata = train_data)
prediction_testdata <- predict(final_test_model, newdata = test_data)
print(prediction_traindata)
print(prediction_testdata)
print(final_train_model)
print(final_test_model)

train_data$math_performance <- factor(train_data$math_performance, levels = levels(prediction_traindata))
conf_train_matrix <- confusionMatrix(table(prediction_traindata, train_data$math_performance))
print(conf_train_matrix)

test_data$math_performance <- factor(test_data$math_performance, levels = levels(prediction_testdata))
conf_test_matrix <- confusionMatrix(table(prediction_testdata, test_data$math_performance))
print(conf_test_matrix)

precision <- conf_train_matrix$byClass["Pos Pred Value"]
print("Precision for train data:")
print(precision)

recall <- conf_train_matrix$byClass["Sensitivity"]
print("Recall for train data :")
print(recall)

precision <- conf_test_matrix$byClass["Pos Pred Value"]
print("Precision for test data:")
print(precision)

recall <- conf_test_matrix$byClass["Sensitivity"]
print("Recall for test data :")
print(recall)



