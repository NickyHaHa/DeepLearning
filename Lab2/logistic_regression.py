import csv
import numpy as np
import matplotlib.pyplot as plt

# Calculate value of n = W^T · X
def Matrix_n(W, X):
    return np.dot(W, X)

# Activation function to get Y_head
def Sigmoid(W, X):
    Y_head = []
    n = Matrix_n(W, X)
    for i in n:
        # Check max and avoid overflow
        if -i > np.log(np.finfo( type(i) ).max):
            Y_head.append(0)
        else:
            Y_head.append(1 / ( 1 + np.exp( -i )))
    return np.array(Y_head)
    # return 1 / ( 1 + np.exp( - Matrix_n(W, X) ))

# Loss function
def CrossEntropy(Y, Y_head):
    eps = 10**-9
    return np.dot(Y, np.log(Y_head+eps)) + np.dot((1 - Y), np.log(1 - Y_head+eps))

# Update W and b
def GradientDescent(W, Y_head, X, Y, LR):
    return W - (LR * np.dot((Y_head - Y), X)) / (X.shape[0])

def Normalization(sigmoid, msg):
    if msg == "normal":
        sigmoid[sigmoid >= 0.5] = 1
        sigmoid[sigmoid < 0.5] = 0
    else:
        sigmoid[sigmoid >= 0.5] = 5
        sigmoid[sigmoid < 0.5] = 2
    return sigmoid

def ModelTest(test, W):
    test_X = test
    test_X = np.hstack( [ test_X, np.ones([ test_X.shape[0], 1 ]) ] )
    Y_head = Sigmoid(W, test_X.T)
    check = Normalization(np.copy(Y_head), "standard")
    np.savetxt("test_ans.csv", check, fmt="%i", delimiter=",", header="ans")
    print("\n...test_ans.csv stored. <<<")

def main():
    # Load csv file to numpy array
    train = np.genfromtxt("train.csv", delimiter=",")
    test = np.genfromtxt("test.csv", delimiter=",")

    # Data preprocessing for first row deletion
    train = np.delete(train, 0, axis=0)
    test = np.delete(test, 0, axis=0)
    # Get array of col 1 to col last
    train_X = train[:,1:]
    train_X = np.hstack( [ train_X, np.ones([ train_X.shape[0], 1 ]) ] )

    # Get array of col 0 and transform to 0 and 1
    train_Y = train[:,0]
    train_Y[train_Y < 3] = 0
    train_Y[train_Y > 3] = 1

    # w0 ~ w783 and b
    # [w0 ~ w783 b]^T * [x0 ~ x783 1]
    W = np.zeros(785)
    LR = 0.01
    Epoch = 1000
    tau = 10**-2
    normal = True
    correct = 0
    loss = 0

    for i in range( Epoch ):
        correct = 0
        Y_head = Sigmoid(W, train_X.T)
        check = Normalization(np.copy(Y_head), "normal")

        for j in range(4000):
            if check[ j ] == train_Y[ j ]:
                correct += 1
        # LR = LR + 1 / (correct + 0.99)
        W = GradientDescent(W, Y_head, train_X, train_Y, LR)
        loss = CrossEntropy(train_Y.T, Y_head)

        if -loss / 4000 < tau:
            print(">>> Shutdown code : Avg of loss was low enough.\n")
            print("Epoch :", i+1)
            print("Weight(include bias) :", W)
            print("Bias :", W[-1])
            print("Learning Rate :", LR)
            print("Training Accuracy :", correct/4000)
            print("Avg Loss = ", -loss/4000)
            normal = False
            break

    if normal:
        print(">>> Shutdown code : All generation completed.\n")
        print("Epoch :", Epoch)
        print("Weight(include bias) :", W)
        print("Bias :", W[-1])
        print("Learning Rate :", LR)
        print("Training Accuracy :", correct/4000)
        print("Final Avg Loss :", -loss/4000)

    ModelTest(test, W)

main()