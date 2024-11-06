import numpy as np
from datasets import load_from_disk
import pickle
import matplotlib.pyplot as plt
import pandas as pd

mnist_dataset = load_from_disk("mnist_dataset")

images = mnist_dataset["train"]["image"]
labels = mnist_dataset["train"]["label"]

images2 = mnist_dataset["test"]["image"]
labels2 = mnist_dataset["test"]["label"]

images_as_vectors = np.array([np.reshape(image, -1) for image in images]) /255.0

labels_as_array = np.array(labels)
labels_as_array = np.eye(10)[labels_as_array]

images_as_vectors_test = np.array([np.reshape(image, -1) for image in images2]) /255.0

labels_as_array_test2 = np.array(labels2)
labels_as_array_test = np.eye(10)[labels_as_array_test2]


def softmax(x):
    x_exp_shifted = np.exp(x - np.max(x))
    return x_exp_shifted / np.sum(x_exp_shifted)

def relu(x):
    return np.maximum(x,0)

def relu_derivative(x):
    return (x >= 0).astype(float)

def forward(X, P):
    hidden_input1 = np.dot(X, P['w1']) + P['b1']
    hidden_output1 = relu(hidden_input1)

    hidden_input2 = np.dot(hidden_output1, P['w2']) + P['b2']
    hidden_output2 = relu(hidden_input2)

    output_input = np.dot(hidden_output2, P['wout']) + P['bout']
    output_output = softmax(output_input)

    result = {
        "g1": hidden_output1,
        "g2": hidden_output2,
        "g3": output_output,
        "t1": hidden_input1,
        "t2": hidden_input2,
    }
    return result
    
def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    a1 = 1/pow(input_size, 0.5)
    weights_hidden1 = np.random.uniform(low=-a1, high=a1, size=(input_size, hidden_size1)) #* 0.01
    bias_hidden1 = np.zeros((1, hidden_size1))
    a2 = 1/pow(hidden_size1, 0.5)
    weights_hidden2 = np.random.uniform(low=-a2, high=a2,size=(hidden_size1, hidden_size2)) #* 0.01
    bias_hidden2 = np.zeros((1, hidden_size2))
    a3 = 1/pow(hidden_size2, 0.5)
    weights_output = np.random.uniform(low=-a3, high=a3, size=(hidden_size2, output_size)) #* 0.01
    bias_output = np.zeros((1, output_size))

    parameters = {
    "w1": weights_hidden1,
    "b1": bias_hidden1,
    "w2": weights_hidden2,
    "b2": bias_hidden2,
    "wout": weights_output,
    "bout": bias_output
    }
    return parameters

def Loss(gl, y):
    loss = -np.dot(y, np.log(gl.T + 0.00001))
    return np.squeeze(loss)

def back_propagation(G, y, p, X):
    dE_dt3 = G['g3'] - y
    dE_dw3 = np.dot(G['g2'].T, dE_dt3)
    dE_db3 = dE_dt3

    dE_dg2 = np.dot(dE_dt3, p['wout'].T)
    dE_dt2 = dE_dg2 * relu_derivative(G['t2'])
    dE_dw2 = np.dot(G['g1'].T, dE_dt2)
    dE_db2 = dE_dt2

    dE_dg1 = np.dot(dE_dt2, p['w2'].T)
    dE_dt1 = dE_dg1 * relu_derivative(G['t1'])
    X = np.array(X)
    X = X.reshape(X.size, 1)
    dE_dw1 = np.dot(X, dE_dt1)
    dE_db1 = dE_dt1

    DW = {
    "dE_dw3": dE_dw3,
    "dE_db3": dE_db3,
    "dE_dw2": dE_dw2,
    "dE_db2": dE_db2,
    "dE_dw1": dE_dw1,
    "dE_db1": dE_db1
    }
    return DW
    
def upgarde_parameters(dw, alpha):
    parameters['w1'] -= alpha * dw['dE_dw1']
    parameters['w2'] -= alpha * dw['dE_dw2']
    parameters['wout'] -= alpha * dw['dE_dw3']

    parameters['b1'] -= alpha * dw['dE_db1']
    parameters['b2'] -= alpha * dw['dE_db2']
    parameters['bout'] -= alpha * dw['dE_db3']

input_size = images_as_vectors.shape[1]
hidden_size1 = 128
hidden_size2 = 128
output_size = 10

parameters = initialize_parameters(input_size, hidden_size1, hidden_size2, output_size)

learning_rate = 0.0001

epochs = 100
c = 0

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(epochs):
    train_loss = []
    correct_train = 0
    for i in range(images_as_vectors.shape[0]):
        G = forward(images_as_vectors[i], parameters)
        loss = Loss(G['g3'], labels_as_array[i])
        train_loss.append(loss)
        pred = np.argmax(G['g3'])
        correct_train += (pred == np.argmax(labels_as_array[i]))
        dw = back_propagation(G, labels_as_array[i], parameters, images_as_vectors[i])
        upgarde_parameters(dw, learning_rate)
    
    train_losses.append(np.mean(train_loss))
    train_acc = correct_train / images_as_vectors.shape[0]
    train_accuracies.append(train_acc)
    
    test_loss = []
    correct_test = 0
    for i in range(images_as_vectors_test.shape[0]):
        G = forward(images_as_vectors_test[i], parameters)
        loss = Loss(G['g3'], labels_as_array_test[i])
        test_loss.append(loss)
        pred = np.argmax(G['g3'])
        correct_test += (pred == np.argmax(labels_as_array_test[i]))
    
    test_losses.append(np.mean(test_loss))
    test_acc = correct_test / images_as_vectors_test.shape[0]
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch+1}, Train Loss: {round(train_losses[-1],4)}, Test Loss: {round(test_losses[-1],4)}, Train Acc: {round(train_acc*100,4)}%, Test Acc: {round(test_acc*100,4)}%')

class_counts = np.zeros(10, dtype=int)
for label in labels_as_array_test2:
    class_counts[label] += 1

confusion_matrix = np.zeros((10, 10), dtype=float)
for i in range(images_as_vectors_test.shape[0]):
    G = forward(images_as_vectors_test[i], parameters)
    predicted_label = np.argmax(G['g3'])
    true_label = labels_as_array_test2[i]
    
    confusion_matrix[true_label, predicted_label] += 1.0 / class_counts[true_label]

confusion_df = pd.DataFrame(confusion_matrix, index=range(10), columns=range(10))

print("Матриця якості класифікації:")
print(confusion_df)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Навчання')
plt.plot(test_losses, label='Тестування')
plt.title('Втрати протягом навчання')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Навчання')
plt.plot(test_accuracies, label='Тестування')
plt.title('Точність протягом навчання')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()

plt.tight_layout()
plt.show()
parameters_file_path = "parameters.pkl"
with open(parameters_file_path, 'wb') as f:
    pickle.dump(parameters, f)