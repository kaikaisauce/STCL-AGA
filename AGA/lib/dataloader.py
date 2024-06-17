import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

def STDataloader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last,
    )
    return dataloader

def normalize_data(data, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        scalar = MinMax01Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    return scalar

def get_dataloader(data_dir, dataset, batch_size, test_batch_size, scalar_type='Standard'):
    # Load the data from a single file and split into train, val, and test sets
    file_path = os.path.join(data_dir, dataset + '_reduced.npz')
    data = np.load(file_path)
    
    # Print the structure of 'data' key to understand how to split x_data and y_data
    full_data = data['data']
    print("Shape of the full data:", full_data.shape)
    
    # Reshape or split the data as needed
    num_samples, num_nodes, num_features = full_data.shape

    # Define the number of time steps
    time_steps = 12

    # Create new data arrays with the desired shape
    x_data = []
    y_data = []

    for i in range(num_samples - time_steps):
        x_data.append(full_data[i:i + time_steps])
        y_data.append(full_data[i + time_steps])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print("Shape of x_data:", x_data.shape)
    print("Shape of y_data:", y_data.shape)

    # Split the data into train (60%), validation (20%), and test (20%)
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


    # Store the split data into a dictionary
    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test
    }

    # Normalize the data
    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)
    
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])

    print("Shape of normalized x_train:", data['x_train'].shape)
    print("Shape of normalized y_train:", data['y_train'].shape)
    print("Shape of normalized x_val:", data['x_val'].shape)
    print("Shape of normalized y_val:", data['y_val'].shape)
    print("Shape of normalized x_test:", data['x_test'].shape)
    print("Shape of normalized y_test:", data['y_test'].shape)

    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'], 
        data['y_train'], 
        batch_size, 
        shuffle=True
    )
    dataloader['val'] = STDataloader(
        data['x_val'], 
        data['y_val'], 
        test_batch_size, 
        shuffle=False
    )
    dataloader['test'] = STDataloader(
        data['x_test'], 
        data['y_test'], 
        test_batch_size, 
        shuffle=False, 
        drop_last=False
    )
    dataloader['scaler'] = scaler
    return dataloader

if __name__ == '__main__':
    loader = get_dataloader('/Users/zdk/Desktop/tj2024spring/0024s时空增强自监督学习/AGA02/data/PEMS04/', 'PEMS04', batch_size=64, test_batch_size=64)
    for key in loader.keys():
        print(key)
