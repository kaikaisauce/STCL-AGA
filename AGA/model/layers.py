import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

from model.aug import sim_global

def node_similarity(flow_data):
    """
    Calculate the node feature heterogeneity measurement.
    
    :param flow_data: tensor, original flow [n, l, v, c]
    :return sim: tensor, node similarity, [n, v, v]
    """
    n, l, v, c = flow_data.shape
    # Calculate mean and variance within the time window
    mean = torch.mean(flow_data, dim=1, keepdim=True)  # [n, 1, v, c]
    variance = torch.var(flow_data, dim=1, keepdim=True)  # [n, 1, v, c]

    # Expand mean and variance to allow broadcasting
    mean = mean.unsqueeze(3)  # [n, 1, v, 1, c]
    variance = variance.unsqueeze(3)  # [n, 1, v, 1, c]
    
    # Permute dimensions to match for subtraction
    mean_diff = mean - mean.permute(0, 1, 3, 2, 4)  # [n, 1, v, v, c]
    variance_diff = variance - variance.permute(0, 1, 3, 2, 4)  # [n, 1, v, v, c]
    
    sigma = torch.std(mean_diff) + torch.std(variance_diff)  # Normalization parameter
    
    similarity = 1 - ((mean_diff ** 2).sum(dim=-1) + variance_diff.sum(dim=-1)) / sigma  # [n, v, v]
    
    return similarity.squeeze(1)  # [n, v, v]

class SpatialContrastLearning(nn.Module):
    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(SpatialContrastLearning, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.tau = tau
        self.d_model = c_in * c_in
        self.batch_size = batch_size
        self.prototypes = nn.Linear(c_in * c_in, nmb_prototype, bias=False)  # Adjusted input dimension
        
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nlvc
        :return loss: contrastive loss
        """
        # Ensure the input shape is [n, l, v, c]
        assert len(z1.shape) == 4, f"Expected input shape [n, l, v, c], but got {z1.shape}"
        assert z1.shape == z2.shape, f"Expected z1 and z2 to have the same shape, but got {z1.shape} and {z2.shape}"
        
        # Compute node similarity
        sim1 = node_similarity(z1)  # [n, v, v]
        sim2 = node_similarity(z2)  # [n, v, v]
        
        # Normalize sim1 and sim2
        sim1_norm = self.l2norm(sim1.view(sim1.size(0), -1))
        sim2_norm = self.l2norm(sim2.view(sim2.size(0), -1))
        
        # Compute prototypes
        zc1 = self.prototypes(sim1_norm)  # [n, v*v] -> [n, nmb_prototype]
        zc2 = self.prototypes(sim2_norm)  # [n, v*v] -> [n, nmb_prototype]
        
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        
        # Compute the contrastive loss
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2
    
@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()

class TemporalContrastLearning(nn.Module):
    '''Temporal heterogeneity modeling in a contrastive manner.
    '''
    def __init__(self, c_in, batch_size, num_nodes, device):
        super(TemporalContrastLearning, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))  # representation weights
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) 
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        
        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        self.lbl = lbl.to(device)
        
        self.n = batch_size

    def forward(self, z1, z2):
        '''
        :param z1, z2 (tensor): shape nlvc, i.e., (batch_size, seq_len, num_nodes, feat_dim)
        :return loss: loss of generative branch. nclv
        '''
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1)  # nlvc->nvc
        s = self.read(h)  # s: summary of h, nc

        # select another region in batch
        idx = torch.randperm(self.n)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)

        # Flatten logits to match the shape of lbl
        batch_size, _, num_nodes, _ = logits.shape
        logits = logits.reshape(batch_size, -1)
        self.lbl = self.lbl.reshape(batch_size, -1)

        loss = self.b_xent(logits, self.lbl)
        return loss

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        '''Apply an average on graph.
        :param h: hidden representation, (batch_size, num_nodes, feat_dim)
        :return s: summary, (batch_size, feat_dim)
        '''
        s = torch.mean(h, dim=1)
        s = self.sigm(s) 
        return s

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)  # similar to score of CPC

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        '''
        :param s: summary, (batch_size, feat_dim)
        :param h_rl: real hidden representation (w.r.t summary),
            (batch_size, num_nodes, feat_dim)
        :param h_fk: fake hidden representation
        :return logits: prediction scores, (batch_size, num_nodes, 2)
        '''
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2) 
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        # Ensure logits shape as (batch_size, 2, num_nodes, 1)
        logits = torch.cat((sc_rl.unsqueeze(1), sc_fk.unsqueeze(1)), dim=1)
        logits = logits[:, :, 0, :, :]  # Remove the redundant time step

        return logits

class STEncoder(nn.Module):
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1):
        super(STEncoder, self).__init__()
        self.Ks = Ks
        c = blocks[0]
        self.tconv11 = TemporalConvLayer(Kt, 3, c[1], "GLU", dilation=2)  # Adjusted input channels to 3 and added dilation
        self.pooler = Pooler(input_length - (Kt - 1), c[1])
        
        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2], dilation=2)
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU", dilation=2)
        
        self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2], dilation=2)
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)
        
        self.s_sim_mx = None
        self.t_sim_mx = None
        
        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU", dilation=2)
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt - 1

    def forward(self, x0, graph):
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)
        
        in_len = x0.size(1)  # x0, nlvc
        if in_len < self.receptive_field:
            x = F.pad(x0, (0, 0, 0, 0, self.receptive_field - in_len, 0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes), nclv 
        
        # ST block 1
        x = self.tconv11(x)  # nclv
        x, x_agg, self.t_sim_mx = self.pooler(x)
        print("Shape of x_agg:", x_agg.shape)
        #x_agg_reshaped = x_agg.permute(0, 2, 1)
        #print("Shape of x_agg_reshaped:", x_agg_reshaped.shape)  # Debug print
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')
        print("Shape of s_sim_mx:", self.s_sim_mx.shape)

        x = self.sconv12(x, Lk)  # nclv
        x = self.tconv13(x)  
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        # ST block 2
        x = self.tconv21(x)
        x = self.sconv22(x, Lk)
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        # out block
        x = self.out_conv(x)  # ncl(=1)v
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1)))  # nlvc
        return x  # nl(=1)vc

    def _cheb_polynomial(self, laplacian, K):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [v, v].
        :return: the multi order Chebyshev laplacian, [K, v, v].
        """
        N = laplacian.size(0)  
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float) 
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    def _cal_laplacian(self, graph):
        """
        return the laplacian of the graph.

        :param graph: the graph structure **without** self loop, [v, v].
        :return: graph laplacian.
        """
        I = torch.eye(graph.size(1), device=graph.device, dtype=graph.dtype)
        print("Shape of graph:", graph.shape)
        print("Shape of I:", I.shape)
        graph = graph + I  # add self-loop to prevent zero in D
        print("Shape of graph:", graph.shape)
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        print("Shape of D:", D.shape)
        L = I - torch.matmul(torch.matmul(D, graph), D)
        return L

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu", dilation=2):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.dilation = dilation
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), dilation=(dilation, 1))
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), dilation=(dilation, 1))

    def forward(self, x):
        """
        :param x: (n, c, l, v)
        :return: (n, c, l-dilation*(kt-1), v)
        """
        padding = (self.kt - 1) * self.dilation
        x_padded = F.pad(x, (0, 0, padding, 0))
        
        x_conv = self.conv(x_padded)
        
        x_in = self.align(x)[:, :, :x_conv.shape[2], :]
        
        if self.act == "GLU":
            result = (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        elif self.act == "sigmoid":
            result = torch.sigmoid(x_conv + x_in)
        else:
            result = torch.relu(x_conv + x_in)
        
        return result

class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.ks = ks
        self.c_in = c_in
        self.c_out = c_out
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # Graph wavelet convolution
        # Compute the wavelet kernel function Ψ_D
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)  
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b 
        
        # Align the input channels if necessary
        x_in = self.align(x) 
        
        # Apply the non-linear activation function
        return torch.relu(x_gc + x_in)

class Pooler(nn.Module):
    '''Pooling the token representations of region time series into the region level.
    '''
    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model 
        """
        super(Pooler, self).__init__()

        # attention matrix
        self.att = FCLayer(d_model, n_query) 
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)  # softmax on the seq_length dim, nclv

        self.d_model = d_model
        self.n_query = n_query 
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')
        
    def forward(self, x):
        x_in = self.align(x)[:, :, -self.n_query:, :]  # ncqv
        # calculate the attention matrix A using key x   
        A = self.att(x)  # x: nclv, A: nqlv 
        A = F.softmax(A, dim=2)  # nqlv

        # calculate region embedding using attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2)  # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg)  # ncv->nvc

        # calculate the temporal similarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2))  # A: lnqv->lnv
        return torch.relu(x + x_in), x_agg.detach(), A.detach()

class MLP(nn.Module):
    def __init__(self, c_in, c_out): 
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2)))  # nlvc->nclv
        x = self.fc2(x).permute(0, 2, 3, 1)  # nclv->nlvc
        return x

class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)  

    def forward(self, x):
        return self.linear(x)
