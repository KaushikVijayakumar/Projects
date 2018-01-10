library ("dplyr")
# check the current working directory
getwd()

# Assign the working directory to the local folder where the file is present 
setwd("D:/Kaushik/Uconn Related/Study and Research/My Projects/Iris")

iris
summary(iris)
str(iris)
head(iris)

skewness(summary)

plot(iris, Sepal.Length, Sepal.Width, col = )

ggplot(iris, aes(x= Sepal.Length, y = Sepal.Width, col = Species)) +
  geom_point() 


ggplot(iris, aes(x= Petal.Length, y = Petal.Width, col = Species)) +
  geom_point() 


boxplot(iris$Sepal.Length ~ iris$Species)
par(mfrow = c(2,2))

dim(iris)


for(i in 1:4)  
  boxplot(iris[,i] ~ iris$Species,data = iris, main = names(iris)[i])

working_data_iqr = iris


#partitioning the data
set.seed(1232)
pd = sample (2, nrow(working_data_iqr), replace = T, prob = c(0.5,0.5))

working_data_iqr_train    = working_data_iqr[pd==1,]
working_data_iqr_validate = working_data_iqr[pd==2,]


working_data_iqr1  = working_data_iqr
working_data_iqr1$Species = NULL





#Store the result and return the value same time using ()
(result = kmeans(working_data_iqr1,3))
result$cluster
result$centers


# plotting the centroids of the data
plot(working_data_iqr$Petal.Length,working_data_iqr$Petal.Width, col = result$cluster)
points (result$centers[,c('Petal.Length', 'Petal.Width')], pch =8,cex = 2, col = 'purple')


# See the confusion matric of the data
table(working_data_iqr$Species,result$cluster)
result$centers[,c('Petal.Length', 'Petal.Width')]



################ DECISION TREE
install.packages("party")
library(party)

tree = ctree(working_data_iqr$Species ~ working_data_iqr$Sepal.Length + working_data_iqr$Sepal.Width + 
               working_data_iqr$Petal.Length + working_data_iqr$Petal.Width)
             #controls = ctree_control(mincriterion = .99,minsplit=500))
plot(tree)

library(rpart)
fit = rpart(working_data_iqr$Species ~ working_data_iqr$Sepal.Length + working_data_iqr$Sepal.Width + 
               working_data_iqr$Petal.Length + working_data_iqr$Petal.Width)

plot(fit)

# predition
working_data_iqr

result_dt = predict(tree, working_data_iqr)
pred_tab = table(working_data_iqr$Species,result_dt)
pred_tab 

# Calculation the misclassification error
(1-sum(diag(pred_tab))/sum(pred_tab))*100

############# Linear Discriminant analysis
lda = lda(working_data_iqr$Species ~ working_data_iqr$Sepal.Length + working_data_iqr$Sepal.Width + 
               working_data_iqr$Petal.Length + working_data_iqr$Petal.Width)

# Remember the $class
result_predict = predict(lda, newdata = working_data_iqr[, 1:4])$class

accuracy = table(result_predict,working_data_iqr[,5])


# Calculation the misclassification error
(1-sum(diag(accuracy))/sum(accuracy))*100



#################### Logistic Regression
Lr = glm(data = working_data_iqr, formula = working_data_iqr$Species ~ working_data_iqr$Sepal.Length + working_data_iqr$Sepal.Width + 
            working_data_iqr$Petal.Length + working_data_iqr$Petal.Width)



?glm





