import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import svm, model_selection, decomposition, metrics


"""
==========================================
Support Vector Machine in order to classify datasets transfusion.tst and pima_indians_diabetes
Creators:
Adam Kałuża s20831
Krzysztof Lewandowski s20491
==========================================
To run program install:
pip install numpy
pip install matplotlib
pip install mlxtend
pip install sklearn
==========================================
Usage:
python svs.py
==========================================
"""


def classify_svc(split_size, C, gamma, data_file, title):
    """
    Trains SVC model on train part of the dataset with received configuration (kernel, c & gamma).
    Classifies and compares the result to the test part.
    """

    data = np.loadtxt(data_file, delimiter=',')
    a, b = data[:, :2], data[:, -1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(a, b.astype(np.int_), test_size=split_size)

    """
    Create the SVC model and start training
    """
    svc = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    svc.fit(X_train, y_train)

    """
    Compare a section of model results with desired outcomes
    """
    y_model_outcome = svc.predict(X_test)

    """
    Reduce dimension whilst keeping the data structure with the goal of visualizing the data on a simple x,y chart
    """
    pca = decomposition.PCA(n_components=2)
    X_train2 = pca.fit_transform(X_train)
    svc.fit(X_train2, y_train)
    plot_decision_regions(X_train2, y_train, clf=svc, legend=2)
    plt.title(title)
    plt.show()

    """
    Calculate the accuracy score.
    """
    return metrics.accuracy_score(y_test, y_model_outcome)


if __name__ == "__main__":
    diabetes_svc_result = classify_svc(0.2, 10, 0.0001, 'pima-indians-diabetes.txt', 'Pima indians diabetes')
    transfusion_svc_result = classify_svc(0.2, 100, 0.0002, 'transfusion.txt', 'Transfusion')
    print(f"Accuracy of SVC model:"
          f"\nPima indians diabetes dataset: {diabetes_svc_result}"
          f"\ntransfusion dataset: {transfusion_svc_result}"
          )
