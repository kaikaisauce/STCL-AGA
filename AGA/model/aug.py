import copy
import numpy as np 
import torch
import torch.nn as nn

class SimplePredictiveModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplePredictiveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_residual(flow_data, model, eta):
    batch_size, seq_len, num_nodes, num_features = flow_data.shape
    residuals = []

    for t in range(1, seq_len):
        F_t_minus_1 = flow_data[:, t-1, :, :].reshape(batch_size * num_nodes, num_features)
        X_t = flow_data[:, t, :, :].reshape(batch_size * num_nodes, num_features)
        X_t_hat = model(F_t_minus_1).reshape(batch_size, num_nodes, num_features)
        H_t = X_t.reshape(batch_size, num_nodes, num_features) - X_t_hat
        F_t_prime = flow_data[:, t, :, :] - eta * H_t
        residuals.append(F_t_prime.unsqueeze(1))

    residuals = torch.cat(residuals, dim=1)
    return residuals

def sim_global(flow_data, sim_type='cos'):
    print(f"Flow data shape: {flow_data.shape}")
    if len(flow_data.shape) == 3:
        n, v, c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1  # calculate 2-norm of each node, dim N
        sim = torch.einsum('nvc, nmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only supports shape length of 3 but got {}.'.format(len(flow_data.shape)))

    print(f"Similarity matrix shape before scaling: {sim.shape}")
    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        #print (sim.shape, scaling.shape)
        sim = sim.unsqueeze(-1) * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5 
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only supports sim_type in [att, cos].')
    
    print(f"Similarity matrix shape after scaling: {sim.shape}")
    return sim



def aug_dama(sim_mx, input_graph, percent=0.2, theta=0.5, alpha=0.5, beta=0.5):
    drop_percent = percent / 2
    
    # Adjusting the adjacency matrix based on similarity threshold
    A_temp = torch.where(sim_mx < theta, input_graph, torch.tensor(0.0))
    
    # Assuming Y is the feature matrix with shape [v, c]
    v = input_graph.shape[0]
    c = sim_mx.shape[-1]  # Adjusted based on given similarity matrix

    # Randomly generating Y for demonstration; replace with actual feature matrix
    Y = torch.randn(v, c)

    # Updating the adjacency matrix
    outer_product = torch.mm(Y, Y.t())
    aug_graph = alpha * A_temp + beta * outer_product

    # Ensure the shape remains the same
    aug_graph = aug_graph[:v, :v]
    
    return aug_graph

def aug_nm(t_sim_mx, flow_data, percent=0.2):
    l, n, v = t_sim_mx.shape
    n, l_flow, v, c = flow_data.shape
    mask_num = int(n * l_flow * v * percent)
    aug_flow = copy.deepcopy(flow_data)

    # Calculate the heterogeneity score
    heterogeneity_score = torch.sum(t_sim_mx, dim=0)  # [v]
    mask_prob = 1 - torch.tanh(heterogeneity_score)  # [v]
    print("Shape of mask_prob:", mask_prob.shape)
    print("Shape of mask_prob.unsqueeze(0):", mask_prob.unsqueeze(0).shape)
    mask_prob = mask_prob.unsqueeze(1).expand(n, l_flow, v)  # [n, l, v]
    print("Shape of t_sim_mx:", t_sim_mx.shape)
    print("Shape of flow_data:", flow_data.shape)
    print("Shape of heterogeneity_score:", heterogeneity_score.shape)
    print("Shape of mask_prob:", mask_prob.shape)

    # Generate a masking matrix based on the calculated probabilities
    bernoulli_dist = torch.bernoulli(mask_prob)  # [n, l, v]
    mask_matrix = 1 - bernoulli_dist  # [n, l, v]
    print("Shape of bernoulli_dist:", bernoulli_dist.shape)
    print("Shape of mask_matrix:", mask_matrix.shape)

    # Apply the mask to the flow data
    aug_flow = aug_flow * mask_matrix.unsqueeze(-1)  # [n, l, v, c]
    print("Shape of aug_flow after masking:", aug_flow.shape)

    return aug_flow



def apply_residual_enhanced_decomposition(flow_data, eta):
    batch_size, seq_len, num_nodes, num_features = flow_data.shape
    input_dim = num_nodes * num_features
    output_dim = num_features
    model = SimplePredictiveModel(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    residuals = calculate_residual(flow_data, model, eta)
    return residuals
