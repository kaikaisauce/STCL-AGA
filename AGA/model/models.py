import torch.nn as nn
import torch

from lib.utils import masked_mae_loss
from model.aug import (
    aug_dama, 
    aug_nm, 
)
from model.layers import (
    STEncoder, 
    SpatialContrastLearning, 
    TemporalContrastLearning, 
    MLP, 
)

class STCLAGA(nn.Module):
    def __init__(self, args):
        super(STCLAGA, self).__init__()
        # spatial temporal encoder
        self.encoder = STEncoder(Kt=3, Ks=3, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)
        
        # traffic flow prediction branch
        self.mlp = MLP(args.d_model, args.d_output)
        # temporal heterogenrity modeling branch
        self.thm = TemporalContrastLearning(args.d_model, args.batch_size, args.num_nodes, args.device)
        # spatial heterogenrity modeling branch
        self.shm = SpatialContrastLearning(args.d_model, args.batch_size, args.shm_temp)
        self.mae = masked_mae_loss(mask_value=5.0)
        self.args = args
    
    def forward(self, view1, graph):
        repr1 = self.encoder(view1, graph) # view1: n,l,v,c; graph: v,v

        s_sim_mx = self.fetch_spatial_sim()
        graph2 = aug_dama(s_sim_mx, graph, percent=self.args.percent*2)
        
        t_sim_mx = self.fetch_temporal_sim()
        view2 = aug_nm(t_sim_mx, view1, percent=self.args.percent)
        
        repr2 = self.encoder(view2, graph2)
        return repr1, repr2

    def fetch_spatial_sim(self):
        return self.encoder.s_sim_mx.cpu()
    
    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx.cpu()

    def predict(self, z1, z2):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        return self.mlp(z1)

    def loss(self, z1, z2, y_true, scaler, loss_weights):
        l1 = self.pred_loss(z1, z2, y_true, scaler)
        sep_loss = [l1.item()]
        loss = loss_weights[0] * l1 

        l2 = self.temporal_loss(z1, z2)
        sep_loss.append(l2.item())
        loss += loss_weights[1] * l2
        
        l3 = self.spatial_loss(z1, z2)
        sep_loss.append(l3.item())
        loss += loss_weights[2] * l3 
        return loss, sep_loss

    def pred_loss(self, z1, z2, y_true, scaler):
        y_pred = self.predict(z1, z2)
        
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
        
        # Ensure y_true and y_pred have the same shape
        y_true = y_true.expand_as(y_pred)

        # Check for NaNs or infinite values in y_pred and y_true
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print("y_pred contains NaNs or infinite values")
        if torch.isnan(y_true).any() or torch.isinf(y_true).any():
            print("y_true contains NaNs or infinite values")
 
        # Calculate the mean absolute error loss
        loss = self.mae(y_pred, y_true)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Loss contains NaNs or infinite values")

        return loss
    
    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)
