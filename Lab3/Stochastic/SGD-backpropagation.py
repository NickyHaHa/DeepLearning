import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class Layer:

    def __init__(self, input_size, output_size, lr):
        self.weight = np.random.uniform(-0.01, 0.01, (output_size, input_size))
        self.bias = np.random.uniform(-0.01, 0.01, output_size)
        # self.weight = np.zeros(shape=(output_size, input_size))
        # self.bias = np.zeros(output_size)
        self.ahead = np.array([])
        self.delta = np.array([])
        self.lr = lr

    def sigmoid(self, n):
        eps = 10**-9
        return np.clip( 1 / ( 1 + np.exp( -n ) ), 0+eps, 1-eps)

    def cross_entropy(self, y):
        eps = 10**-9
        return np.dot(y, np.log(self.ahead+eps)) + np.dot((1-y), np.log(1-self.ahead+eps))

    def get_weights(self):
        return self.weight, self.bias

    def get_delta(self):
        return self.delta

    def get_ahead(self):
        return self.ahead
    
    # Calculate Y ahead
    def cal_ahead(self, a):
        n = self.weight.dot(a) + self.bias
        # print("n shape:", n.shape)
        self.ahead = self.sigmoid(n)

        return self.ahead

    # The last delta (error vector)
    def cal_error_vector(self, y):
        self.delta = self.ahead - y

        return self.delta

    # Delta = left-side product right-side with element wise 
    def cal_delta(self, weight, delta):
        l = weight.T.dot(delta)
        r = np.multiply(self.ahead, 1-self.ahead)
        self.delta = np.multiply(l, r)

        return self.delta

    # W = w - lr * delta * a.T
    # B = b - lr * delta
    def update_weights(self, a):
        # Add new axis on 1d ndarray and make multiplication
        delta, pre_a = self.delta[:, None], a[:, None]
        r = self.lr * delta.dot(pre_a.T)
        self.weight = self.weight - r

        r = self.lr * self.delta
        self.bias = self.bias - r

# Data preprocessing
def PreProcessing(train, test):
    # First row deletion
    train = train.to_numpy()
    test = test.to_numpy()

    # Split data to training and validation
    split = int( len(train) * 0.9 )
    validation = train[ split : ]   # 1600, 785
    train = train[ : split ]        # 14400, 785

    trainX = train[ :,1: ]          # 14400, 784
    trainY = train[ :,0 ]           # 14400,   
    valX = validation[ :,1: ]
    valY = validation[ :,0 ]

    # One-hot encoding
    trainY[ trainY == 0 ] = 0
    trainY[ trainY == 3 ] = 1
    trainY[ trainY == 8 ] = 2
    trainY[ trainY == 9 ] = 3
    trainY = trainY.astype(int)
    trainY = np.eye(4)[ trainY ]

    valY[ valY == 0 ] = 0
    valY[ valY == 3 ] = 1
    valY[ valY == 8 ] = 2
    valY[ valY == 9 ] = 3    
    valY = valY.astype(int)
    valY = np.eye(4)[ valY ]

    return trainX, trainY, valX, valY, test

def Feedforward(x, layers):
    ahead = x
    for l in layers:
        ahead = l.cal_ahead(ahead)
        w, b = l.get_weights()
        # print("W shape: ", w.shape)
        # print("B shape: ", b.shape)
        # print("A shape: ", ahead.shape)
        # print(ahead)       
    # print(layers[ -1 ].get_ahead())

def Backpropagate(layer_cnt, layers):
    lc = layer_cnt - 2
    while lc >= 0:
        next_w = layers[ lc+1 ].get_weights()[ 0 ]
        next_delta = layers[ lc+1 ].get_delta()
        layers[ lc ].cal_delta(next_w, next_delta)
        # print("D shape: ", layers[ lc ].get_delta().shape)
        # print(layers[ lc ].get_delta())

        lc -= 1

def Update(layer_cnt, layers, x):
    layers[ 0 ].update_weights(x.T)
    w, b = layers[ 0 ].get_weights()
    # print("\n-Update-\n")
    # print("W shape: ", w.shape)
    # print(w)
    # print("B shape: ", b.shape)
    # print(b)


    lc = 1
    while lc < layer_cnt:
        pre_ahead = layers[ lc-1 ].get_ahead()
        layers[ lc ].update_weights(pre_ahead)
        w, b = layers[ lc ].get_weights()
        # print("W shape: ", w.shape)
        # print(w)
        # print("B shape: ", b.shape)
        # print(b)
        lc += 1

def Backward(layer_cnt, layers, pixel, label):
    # lc = layer_cnt - 1
    layers[ -1 ].cal_error_vector(label)
    # print("E shape: ", error_vector.shape)
    # print(error_vector)

    Backpropagate(layer_cnt, layers)
    Update(layer_cnt, layers, pixel)

def Draw(x, trainY, valY, title):
    plt.plot(x, trainY, '-', color='#EA0000', label="T r a i n "+title)
    plt.plot(x, valY, '-', color='#0080FF', label="V a l i d a t e "+title)
    plt.xlabel("E p o c h")
    plt.ylabel(title)
    plt.legend()
    plt.savefig(title+".png")
    # plt.show()

def ModelTest(x, layers):
    x = x / 255.0
    # print(x.shape)
    ans = np.array([])
    for i in range( x.shape[0] ):
        Feedforward(x[i], layers)
        guess = layers[ -1 ].get_ahead()
        tmp = np.argmax(guess)

        if i % 1000 == 0:
            print('.')
        if tmp == 0:
            ans = np.append(ans, 0)
        elif tmp == 1:
            ans = np.append(ans, 3)
        elif tmp == 2:
            ans = np.append(ans, 8)
        else:
            ans = np.append(ans, 9)
    np.savetxt("test_ans.csv", ans, fmt="%i", delimiter=",", header="ans")
    print("...test_ans.csv stored. <<<")

def main():
    # Load csv file ( fashion_mnist (28, 28, 1) )
    train = pd.read_csv("lab3_train.csv")
    test = pd.read_csv("lab3_test.csv")

    trainX, trainY, valX, valY, test = PreProcessing(train, test)

    LR, Epoch, tau = 0.01, 50, 10**-2
    normal = True
    layers = np.array([])
    layers = np.append(layers, Layer(784, 16, LR))
    layers = np.append(layers, Layer(16, 4, LR))
    layer_cnt = len(layers)
    print(f'{layer_cnt}-Layer: 784, 16, 4')

    pixel = np.copy(trainX) / 255.0
    label = np.copy(trainY)
    pixel_v = np.copy(valX) / 255.0
    label_v = np.copy(valY)

    # print("X shape: ", pixel[0].shape)
    # print("Y shape: ", label[0].shape)

    train_acc, train_loss = np.array([]), np.array([])
    val_acc, val_loss = np.array([]), np.array([])

    for e in range( Epoch ):
        
        train_correct, val_correct = 0, 0
        train_err, val_err = 0, 0

        # Training
        for i in range( pixel.shape[0] ):
            Feedforward(pixel[ i ], layers)

            guess = np.argmax(layers[ -1 ].get_ahead())
            real = np.argmax(label[ i ])

            if guess == real:
                train_correct += 1

            train_err += np.sum(layers[ -1 ].cross_entropy(label[ i ])) / 4
            
            Backward(layer_cnt, layers, pixel[ i ], label[ i ])

        acc = train_correct / pixel.shape[0]
        loss = -train_err / pixel.shape[0]
        train_acc = np.append(train_acc, acc)
        train_loss = np.append(train_loss, loss)

        # Validation
        for j in range( pixel_v.shape[0] ):
            Feedforward(pixel_v[ j ], layers)
            guess = np.argmax(layers[ -1 ].get_ahead())
            real = np.argmax(label_v[ j ])
            if guess == real:
                val_correct += 1
            
            val_err += np.sum(layers[ -1 ].cross_entropy(label_v[ j ])) / 4
        
        acc_v = val_correct / pixel_v.shape[0]
        loss_v = -val_err / pixel_v.shape[0]
        val_acc = np.append(val_acc, acc_v)
        val_loss = np.append(val_loss, loss_v)

        # if (e+1) % 5 == 0:
        print(f'== {e+1} / {Epoch} == train-acc: {acc*100:.4f}, validate-acc: {acc_v*100:.4f}, train-loss: {loss:.6f}, validate-loss: {loss_v:.6f}')
        # print(f'== {e+1} / {Epoch} ==')
        # if acc_v < acc:
        #     break

        if loss < tau or loss_v < tau:
            print("\n>>> Shutdown code : Avg of loss was low enough.\n")
            print("Epoch: ", e+1)
            print("Learning Rate: ", LR)
            print(f'Training accuracy: {acc:.3f}')
            print(f'Training loss: {loss:.5f}')
            print(f'Validation accuracy: {acc_v:.3f}')
            print(f'Validation loss: {loss_v:.5f}')
            normal = False
            break

    if normal:
        print("\n>>> Shutdown code : All generation completed.\n")
        print("Epoch: ", Epoch)
        print("Learning Rate: ", LR)
        print(f'Training accuracy: {train_correct/pixel.shape[0]:.3f}')
        print(f'Training loss: {-train_err/pixel.shape[0]:.5f}')
        print(f'Validation accuracy: {val_correct/pixel_v.shape[0]:.3f}')
        print(f'Validation loss: {-val_err/pixel_v.shape[0]:.5f}')

    Draw(np.linspace(0, Epoch, Epoch), train_acc, val_acc, "-A c c")
    plt.clf()
    Draw(np.linspace(0, Epoch, Epoch), train_loss, val_loss, "-L o s s")

    ModelTest(test, layers) 

main()