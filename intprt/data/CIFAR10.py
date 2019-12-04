import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10():
    train = list()
    for i in range(5):
        # C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data
        obj = unpickle(f'data/CIFAR-10/data_batch_{i+1}')
        train.append(obj)

    test = unpickle(f'data/CIFAR-10/test_batch')

    X = train[0][b'data']
    y = train[0][b'labels']
    y = np.asarray(y)

    X_test =  test[b'data']
    y_test = test [b'labels']
    y_test = np.asarray(y_test)

    for i in range (1,len(train)):
        X = np.vstack((X, train[i][b'data']))
        y = np.hstack((y, train[i][b'labels']))

    # transpose the colour axis to be the inner-most one, giving the image format (H, W, C)
    X = X.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    X_test = X_test.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    return X, y, X_test, y_test

## To test [OLD]:
# train, test = load_CIFAR10()
# n = 5
# print(f'First {n} records in...')
# for i in range(len(train)):
#     print(f'batch {i+1}:')
#     obj_data = train[i][b'data']
#     obj_labels = train[i][b'labels']
#     print(f'{obj_data[:n,:n]}')
#     print(f'{obj_labels[:n]}')
# print(f'Test data:')
# obj_data = test[b'data']
# obj_labels = test[b'labels']
# print(f'{obj_data[:n,:n]}')
# print(f'{obj_labels[:n]}')
