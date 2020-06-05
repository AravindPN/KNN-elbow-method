# KNN-elbow-method
Python function to find out the best K value while performing K-Nearest Neighbor Analysis, using the elbow method.

What the elbow method does is to perform KNN predictions for different K values on the test data and find out the error rate.
The K value with the least error will be giving us the best predictions.

Since this has to be done whenever you perform a KNN analysis and there is no built in function in sklearn for this, you can just import this file and use this in your program!

After the train test split of your data, you will obtain the X_train, X_test, y_train and y_test data. (These are the terms used in the train_test_split function's documentation).
All you have to do is to pass in these data as parameters for the elbow function.
You should also pass in the lower_k and upper_k value, so that the function will check the error rates of the K values in the range(lower_k, upper_k + 1).
The function will then print out the K value with the least error.

Additional features:
- get_predictions parameter: When set to True, the function will return the predicted values based on the provided data using the best K     value. You can use this predicted values to analyze the confusion matrix or the classification report etc.
- show_plot parameter:  When set to True, the function will display a plot (K values vs. Error rates). You can use this plot for futher     analysis and could choose a K value that you want according to the plot!
