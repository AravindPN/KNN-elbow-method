from sklearn.neighbors import KNeighborsClassifier
from numpy import mean, array
import matplotlib.pyplot as plt


def elbow(X_train, X_test, y_train, y_test, lower_k, upper_k, get_predictions=False, show_plot=False):
    """ This function uses the elbow method to find out the K value with the least error based on your data """

    error_rate = []
    for i in range(lower_k, upper_k + 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        error_rate.append(mean(predictions != y_test))
    best_k = lower_k + array(error_rate).argmin()
    print(f'K value with least error is: {best_k}')
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.xlabel('K values')
        plt.ylabel('Error values')
        plt.title('K values vs. Error rates')
        plt.plot(range(lower_k, upper_k + 1), error_rate, marker='o', markerfacecolor='green', markersize=10)
        plt.show()
        
    if get_predictions:
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        return predictions
