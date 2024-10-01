import torch
import torch.nn as nn

from .encoders import ResNet_LinearRec, RecResNet
OPT_PROJECTORS = {'ResNet_LinearRec' : ResNet_LinearRec, 'RecResNet':RecResNet}
OPT_SIMILARITIES = {'cosine': nn.CosineSimilarity}

class RJEPA(nn.Module):

    def __init__(self, projector_name, cfg_projector, pred_dim=512, similarity = 'cosine', forwardprop = False):

        super(RJEPA, self).__init__()

        self.encoder = OPT_PROJECTORS[projector_name](**cfg_projector, forwardprop=forwardprop)
        dim_proj = self.encoder.dim_proj
        # ------------------------------------------------------------------------

        if dim_proj > pred_dim:
            print('Error: pred_dim debe ser mayor que dim_proj')
            exit()

        self.predictor = nn.Sequential(nn.Linear(dim_proj, pred_dim, bias=False),
                                        nn.LayerNorm(pred_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(pred_dim, dim_proj))

        # Initialization predictor approx identity ------------------------------
        dg_mtx = torch.diag(torch.full((dim_proj,), torch.sqrt(torch.tensor(0.5))))
        rnd_mtx = torch.randn(pred_dim-dim_proj,dim_proj) * torch.sqrt(torch.tensor(0.5/(1.0*(pred_dim-dim_proj))))

        simclr_w = torch.cat((dg_mtx, rnd_mtx), dim=0)

        self.predictor[0].weight = nn.Parameter(simclr_w,requires_grad=True)
        self.predictor[3].weight = nn.Parameter(simclr_w.T,requires_grad=True)
        self.predictor[3].bias.data.zero_()

        # ------------------------------------------------------------------------
        self.similarity = OPT_SIMILARITIES[similarity](dim=1)

        # -----------------------------------------------------------------------

    def forward(self, seq_patches):
        num_patches = len(seq_patches)

        state =  [None for hh in range(self.encoder.num_layers+1)]
        memory =  [None for hh in range(self.encoder.num_layers+1)]
        grads =  [{nn: None for nn, _ in self.encoder.layers_list[hh].rnn_dynamic.named_parameters()} 
                            for hh in range(self.encoder.num_layers)] + [{nn: None for nn, _ in self.encoder.rnn_high.named_parameters()}]

        for t in range(self.encoder.num_layers+1):
            state, memory, grads = self.encoder(seq_patches[t],state, memory, grads)
        
        seq_pred = []
        seq_loss = []
        TotalLoss = 0.0

        for t in range(self.encoder.num_layers+1,num_patches):
            post_patch = seq_patches[t]
            pred = self.predictor(state[-1])
            state, memory, grads = self.encoder(post_patch,state, memory, grads)
            target = state[-1].detach()
            loss = 1-self.similarity(pred,target)

            seq_pred.append(pred)
            seq_loss.append(loss)

        meanloss = sum(seq_loss)/len(seq_loss)
        seq_loss_items = [ll.item() for ll in seq_loss]

        return seq_pred, seq_loss_items, meanloss