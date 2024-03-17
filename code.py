import numpy as np
from sklearn.datasets import load_svmlight_file
import sklearn.model_selection as ms
import os
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------------------
# functions
def sigmoid(n):
    return 1 / (1 + np.exp(-n))


def spamIRLS(X, y, Xt, yt, lmbda):
    w = np.zeros(X.shape[1])
    prev = 0
    tra = 1
    n = 0
    while abs(tra - prev) > 0:
        n += 1
        prev = tra
        w = stepIRLS(X, y, w, i)
        tra = accuracy(X, y, w)
        tea = accuracy(Xt, yt, w)
    print(w)
    return (n, tra, tea, np.linalg.norm(w), evaluateLL(X, y, w, lmbda))


def stepIRLS(X, y, w, lmbda=0):
    p, q = X.shape
    mu = sigmoid(X.dot(w))
    R = mu * (1 - mu)
    error = y - mu
    grad = X.T.dot(error) - lmbda * w
    RX = X * R[:, np.newaxis]
    H = -X.T @ RX - lmbda * np.eye(q)
    return w - (np.linalg.pinv(H) @ grad).ravel()


def accuracy(X, y, w):
    p, q = X.shape
    predicted = (X.dot(w.reshape(q, 1)) > 0).astype(float)
    return np.mean(predicted.ravel() == y)


def evaluateLL(X, y, w, lmbda):
    t = 0
    wex = X.dot(w).ravel()
    t = y * wex - np.log(1 + np.exp(wex))
    return np.sum(t) - lmbda / 2 * (np.linalg.norm(w) ** 2)


# ---------------------------------------------------------------------------------------------
# data loading
p = os.path.dirname(os.path.abspath(__file__))

X_train, y_train = load_svmlight_file(p + "\\a9a Data\\a9a")
X_test, y_test = load_svmlight_file(p + "\\a9a Data\\a9a.t")

y_train = (y_train > 0).astype(float)
y_test = (y_test > 0).astype(float)

X_train = X_train.toarray()
p, q = X_train.shape
XTrain = np.append(np.ones((p, 1)), X_train, axis=1)

X_test = X_test.toarray()
p = X_test.shape[0]
XTest = np.append(np.ones((p, 1)), X_test, axis=1)
XTest = np.append(XTest, np.zeros((p, 1)), axis=1)

# print(np.linalg.det(XTrain.T @ XTrain))
# print(np.linalg.det(XTest.T @ XTest))

# for i in range(XTrain.shape[1]):
#     XTrain[:,i]=XTrain[:,i]-np.mean(XTrain[:,i])
#     XTest[:,i]=XTest[:,i]-np.mean(XTest[:,i])

# print(np.linalg.det(XTrain.T @ XTrain))
# print(np.linalg.det(XTest.T @ XTest))

# worth = np.array(XTrain.sum(axis=0).astype(int)).reshape(124, 1)
# XTrain = XTrain[:, (worth > 10000).ravel()]
# XTest = XTest[:, (worth > 10000).ravel()]

tras = []
teas = []
steps = []
lambdas = []

r = range(0, 5)
# r=np.arange(0, 5, 0.1).tolist()


# # ---------------------------------------------------------------------------------------------
# # testing different lambda values
for i in r:
    n, tra, tea, l2, ll = spamIRLS(XTrain, y_train, XTest, y_test, i)
    steps.append(n)
    tras.append(tra)
    teas.append(tea)
    lambdas.append(i)
    if i > 0:
        print(
            "Lambda=%d, Training Acc=%f, Test Acc=%f, L2 Norm of w=%f, log likelihood=%f. Convergence in %d steps"
            % (i, tra, tea, l2, ll, n)
        )
    else:
        print(
            "Non-Regularised. Training Acc=%f, Test Acc=%f, L2 Norm of w=%f, log likelihood=%f. Convergence in %d steps\n\n"
            % (tra, tea, l2, ll, n)
        )

k = teas.index(max(teas))
l = tras.index(max(tras))

print("\nThe maximum was lambda=%d, convergence=%d steps, training acc=%f, test acc=%f" % (lambdas[k], steps[k], tras[k], teas[k]))
print(
    "and the maximum training accuracy was at lambda=%d, convergence=%d steps, training acc=%f, test acc=%f"
    % (lambdas[l], steps[l], tras[l], teas[l])
)

plt.plot(lambdas, teas, "r", label="Test Accuracy")
plt.plot(lambdas, tras, "b", label="Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Lambda")
plt.legend(loc="upper right")
plt.show()


# ---------------------------------------------------------------------------------------------
# cross validation
# kf = ms.KFold(10, shuffle=True, random_state=0)
# testResults = []
# trainResults = []
# lambdaRange = np.arange(1, 101, 1).tolist()
# # lambdaRange = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,2,3,4,5,6,7,8,9,10])


# # XTrain = np.append(XTrain, XTest, axis=0)
# # y_train = np.append(y_train, y_test, axis=0)

# for i in lambdaRange:
#     tras = []
#     teas = []
#     for trIndices, teIndices in kf.split(XTrain):
#         x, y = XTrain[trIndices], y_train[trIndices]
#         xtest, ytest = XTrain[teIndices], y_train[teIndices]
#         n, tra, tea, l2, ll = spamIRLS(x, y, xtest, ytest, i)
#         teas.append(tea)
#         tras.append(tra)
#     mteas = np.mean(teas)
#     mtras = np.mean(tras)
#     print("Accuracy for Lambda = %f is %f" % (i, mteas))
#     testResults.append(mteas)
#     trainResults.append(mtras)

# m = testResults.index(max(testResults))
# print("10-Fold cross-validation suggests we use lambda=%f" % lambdaRange[m])

# plt.plot(lambdaRange, trainResults, 'b', label="Training Accuracy")
# plt.plot(lambdaRange, testResults, 'r', label="Test Accuracy")
# plt.ylabel('Accuracy')
# plt.xlabel('Lambda')
# plt.legend(loc='upper right')
# plt.show()


# x = np.arange(len(lambdaRange))
# width = 0.35

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, trainResults, width, label='Train Accuracy')
# rects2 = ax.bar(x + width/2, testResults, width, label='Test Accuracy')
# ax.set_ylabel('Accuracy')
# ax.set_xlabel('Lambda')
# ax.set_xticks(x)
# ax.set_xticklabels(lambdaRange)
# ax.legend()
# ax.set_ylim([0.846, 0.85])

# plt.show()

# mmat = np.append(np.array(testResults).reshape(len(lambdaRange), 1),
#                  np.array(lambdaRange).reshape(len(lambdaRange), 1), axis=1)

# mmat = mmat[np.argsort(mmat[:, 0])]

# print(mmat)


# teas = []
# steps = [0]

# w = np.zeros(XTrain.shape[1])
# teas.append(accuracy(XTest, y_test, w))
# prev = 0
# tra = 1
# n = 0
# while(abs(tra-prev) > 0):
#     n += 1
#     steps.append(n)
#     prev = tra
#     w = stepIRLS(XTrain, y_train, w, 0)
#     tra = accuracy(XTrain, y_train, w)
#     teas.append(accuracy(XTest, y_test, w))

# plt.plot(steps[1:], teas[1:], 'c', label="Lambda=0")

# print(teas)

# oldn=n

# teas = []

# w = np.zeros(XTrain.shape[1])
# teas.append(accuracy(XTest, y_test, w))
# prev = 0
# tra = 1
# while(n > 0):
#     n -= 1
#     w = stepIRLS(XTrain, y_train, w, 54)
#     teas.append(accuracy(XTest, y_test, w))

# print(teas)

# plt.plot(steps[1:], teas[1:], 'm', label=("Lambda=%d" % 54))

# teas = []

# w = np.zeros(XTrain.shape[1])
# teas.append(accuracy(XTest, y_test, w))
# prev = 0
# tra = 1
# n=oldn
# while(oldn > 0):
#     oldn -= 1
#     w = stepIRLS(XTrain, y_train, w, 4)
#     teas.append(accuracy(XTest, y_test, w))

# print(teas)

# plt.plot(steps[1:], teas[1:], 'r', label=("Lambda=%d" % 4))

# teas = []

# w = np.zeros(XTrain.shape[1])
# teas.append(accuracy(XTest, y_test, w))
# prev = 0
# tra = 1
# while(n > 0):
#     n -= 1
#     w = stepIRLS(XTrain, y_train, w, 184)
#     teas.append(accuracy(XTest, y_test, w))

# print(teas)

# plt.plot(steps[1:], teas[1:], 'k', label=("Lambda=%d" % 184))


# plt.ylabel('Accuracy')
# plt.xlabel('Iterations')
# plt.legend(loc='lower right')
# plt.show()
