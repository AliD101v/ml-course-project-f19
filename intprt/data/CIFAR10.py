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
        obj = unpickle(f'C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/CIFAR-10/data_batch_{i+1}')
        train.append(obj)
    test = unpickle(f'C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/CIFAR-10/test_batch')

    return train, test

## To test:
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
