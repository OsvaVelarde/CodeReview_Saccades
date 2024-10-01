from torch import tanh
from torch import tensor
import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3

# ------------------------------------------------------------------

class non_dynamic_cell(nn.Module):
    def __init__(self, inp_channels, mm_channels, st_channels, name=None):
        super(non_dynamic_cell, self).__init__()

    def set_forward(self, forwardprop):
        print('NonDynamicCell')

    def forward(self,input_dyn,internal_state,internal_memory, grads):
        return input_dyn, None, None

# ------------------------------------------------------------------

class time_decay_cell(nn.Module):
    def __init__(self, inp_channels, mm_channels, st_channels, name=None):
        super(time_decay_cell, self).__init__()
        self.temporal_parameter = nn.Parameter(tensor([0.0]))

    def set_forward(self,forwardprop):
        self.forward = self._forward_w_grad if forwardprop else self._forward_wo_grad

    def _forward_wo_grad(self,input_dyn,internal_state,internal_memory, grads):
        new_state  = self.temporal_parameter * internal_state + input_dyn if internal_state is not None else input_dyn
        return new_state, None, None

    def _forward_w_grad(self,input_dyn,internal_state,internal_memory, grads):
        new_grads['temporal_parameter'] = None
        new_state =  input_dyn

        if internal_state is not None:
            new_state += self.temporal_parameter * internal_state

            if grads['temporal_parameter'] is not None:
                new_grads['temporal_parameter'] =  self.temporal_parameter * grads['temporal_parameter'] + internal_state

        return new_state, None, new_grads

# ------------------------------------------------------------------

class op_rgcell(nn.Module):
    def __init__(self,in_channels, out_channels, name):
        super(op_rgcell, self).__init__()
        self.operator = conv3x3(in_channels, out_channels) if name == 'conv' else nn.Linear(in_channels, out_channels,bias=False)
        self.name = name
        nn.init.zeros_(self.operator.weight)

    def forward(self, X):
        out = self.operator(X)
        return out

# ------------------------------------------------------------------

class recipgated_cell(nn.Module):
    def __init__(self, inp_channels, mm_channels, st_channels, name='conv'):
        super(recipgated_cell, self).__init__()

        mm_channels = inp_channels
        st_channels = inp_channels

        self.op_tau_mm = op_rgcell(mm_channels,mm_channels,name) 
        self.op_gat_mm = op_rgcell(st_channels,mm_channels,name) 
        self.op_tau_st = op_rgcell(st_channels,st_channels,name) 
        self.op_gat_st = op_rgcell(mm_channels,st_channels,name) 

    def gates(self,internal_state,internal_memory):
        tau_mm = tanh(self.op_tau_mm(internal_memory))
        tau_st = tanh(self.op_tau_st(internal_state ))
        gat_mm = tanh(self.op_gat_mm(internal_state ))
        gat_st = tanh(self.op_gat_st(internal_memory))

        return tau_mm, tau_st, gat_mm, gat_st

    def set_forward(self,forwardprop):
        self.forward = self._forward_w_grad if forwardprop else self._forward_wo_grad

    def _forward_wo_grad(self,input_dyn,internal_state,internal_memory, grads={}):

        if internal_state is not None:
            tau_mm, tau_st, gat_mm, gat_st = gates(internal_state,internal_memory)
            print(tau_mm.shape, tau_st.shape, gat_mm.shape, gat_st.shape, internal_state.shape, internal_memory.shape)
            new_memory = tau_mm * internal_memory + (1 - gat_mm) * input_dyn
            new_state  = tau_st * internal_state  + (1 - gat_st) * input_dyn
            return new_state, new_memory, None
        else:
            return input_dyn, input_dyn, None

    def _forward_w_grad(self,input_dyn,internal_state,internal_memory, grads={}):

        if internal_state is not None:
            tau_mm, tau_st, gat_mm, gat_st = gates(internal_state,internal_memory)
            print(tau_mm.shape, tau_st.shape, gat_mm.shape, gat_st.shape, internal_state.shape, internal_memory.shape)
            exit()
            new_memory = tau_mm * internal_memory + (1 - gat_mm) * input_dyn
            new_state  = tau_st * internal_state  + (1 - gat_st) * input_dyn
            return new_state, new_memory, None
        else:
            return input_dyn, input_dyn, None



# ------------------------------------------------------------------


# gradient rcell
# tau_grad  = (1 - tau_st ** 2) (.) internal_state (.) diag(self.op_tau_st.weights) + tau_st
# input_mu  = (1 - tau_st ** 2) (.) internal_state * internal_state^T
# input_rho = (1 - gat_st ** 2) (.) input_dyn * internal_memory^T 
# mu = tau_grad (.) mu + input_mu
# rho = tau_grad (.) mu + input_rho 


# self.set_forward(bio_computation)
    # def set_forward(self,bio_computation):
    #     if bio_computation:
    #         self.forward = self._forward_bio
    #     else:
    #         self.forward = self._forward_nobio