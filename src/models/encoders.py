from torch import flatten
import torch.nn as nn
import collections as col

from .layers import BasicBlockV1, BasicBlockV2, Bottleneck
OPTS_BLOCK = {'V1':BasicBlockV1,'V2':BasicBlockV2, 'Bottleneck':Bottleneck}

from .rnn_cells import non_dynamic_cell, time_decay_cell, recipgated_cell
OPTS_RNN_CELLS = {'non_rnn': non_dynamic_cell, 'time_decay': time_decay_cell, 'rgated': recipgated_cell}

import torch
# from torchvision.ops.misc import FrozenBatchNorm2d

class ResNet_LinearRec(nn.Module):

    # ============================================================
    # ============================================================

    def __init__(self,input_channels=3, num_filters=64,
                first_layer_kernel_size=7, first_layer_conv_stride=2,first_layer_padding=3,
                first_pool_size=3, first_pool_stride=2, first_pool_padding=1,
                block_fn='Bottleneck', blocks_per_layer=[3, 4, 6, 3], block_strides = [1,2,2,2],growth_factor=2,
                dim_proj=128,rnn_cell='time_decay'):

        super(ResNet_LinearRec, self).__init__()

        self.typeBN = nn.BatchNorm2d

        # Modules of 1st-Stage -----------------------------------
        self.inplanes = num_filters
        current_num_filters = num_filters

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False)

        self.bn1 = self.typeBN(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding)

        # Modules of Layers --------------------------------------
        self.num_layers = len(blocks_per_layer)
        self.layers_list = nn.ModuleList()
        self.block = OPTS_BLOCK[block_fn]

        for (num_blocks, stride) in zip(blocks_per_layer, block_strides):
            channels = current_num_filters * self.block.expansion
            layer = self._make_layer(
                block=self.block,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride)

            self.layers_list.append(layer)
            current_num_filters *= growth_factor

        # Modules of Pooling -------------------------------------        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Modules of Recurrence in Internal Representation -------

        self.head= nn.Sequential(nn.Linear(channels, channels, bias=False),
                                        nn.LayerNorm(channels),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(channels, channels, bias=False),
                                        nn.LayerNorm(channels),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(channels, dim_proj, bias=False),
                                        nn.LayerNorm(dim_proj, elementwise_affine=False),
                                        )

        self.rnn = OPTS_RNN_CELLS[rnn_cell](dim_proj,dim_proj,dim_proj,name='linear')
        self.dim_proj = dim_proj

    # ============================================================
    # ============================================================

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), 
                        self.typeBN(planes * block.expansion))

        layers_ = [('0',block(self.inplanes, planes, stride, downsample, BN=self.typeBN))]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers_.append((str(i),block(self.inplanes, planes, BN=self.typeBN)))
      
        return nn.Sequential(col.OrderedDict(layers_))

    # ============================================================
    # ============================================================

    def forward(self, x, internal_state, memory):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for layer in self.layers_list:
            x = layer(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.head(x)
        internal_state, memory = self.rnn(x,internal_state,memory)
        # grad_m, grad_r = grad_time_decay(prev_m, prev_r, prev_state, input, weight)

        return internal_state, memory #grad_, grad_r


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

class RecResNet(nn.Module):

    # ============================================================
    # ============================================================

    def __init__(self,input_channels=3, num_filters=64,
                first_layer_kernel_size=7, first_layer_conv_stride=2,first_layer_padding=3,
                first_pool_size=3, first_pool_stride=2, first_pool_padding=1,
                block_fn='Bottleneck', blocks_per_layer=[3, 4, 6, 3], block_strides = [1,2,2,2],growth_factor=2,
                dim_proj=128,
                feedback_connections = [],
                rnn_cell_low = ['time_decay','time_decay','time_decay','time_decay'],
                rnn_cell_high = 'time_decay',
                forwardprop=False):
                # frozenBN

        super(RecResNet, self).__init__()

        self.typeBN = nn.BatchNorm2d
        #self.typeBN = FrozenBatchNorm2d if frozenBN else nn.BatchNorm2d

        # Modules of 1st-Stage -----------------------------------
        self.inplanes = num_filters
        current_num_filters = num_filters

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False)

        self.bn1 = self.typeBN(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding)

        # Modules of Layers --------------------------------------
        self.num_layers = len(blocks_per_layer)
        self.layers_list = nn.ModuleList()
        self.block = OPTS_BLOCK[block_fn]

        # Construction -------------------------------------------    
        idx_layer = 0
        
        for (num_blocks, stride) in zip(blocks_per_layer, block_strides):
            channels = current_num_filters * self.block.expansion

            ff_layer = self._make_layer(
                block=self.block,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride)

            feedbacks_per_layer = [
                (connection, int(channels * growth_factor**(connection[0]-connection[1]))) 
                for connection in feedback_connections if connection[1]==idx_layer]

            rnn_layer = RNN_layer(fforward_stage = ff_layer,
                                  feedbacks = feedbacks_per_layer,
                                  rnn_cell=rnn_cell_low[idx_layer],
                                  forwardprop=forwardprop)

            self.layers_list.append(rnn_layer)

            current_num_filters *= growth_factor
            idx_layer += 1

        # Modules of Pooling -------------------------------------        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Modules of Recurrence in Internal Representation -------
        self.dim_proj = dim_proj

        self.head= nn.Sequential(nn.Linear(channels, channels, bias=False),
                                        nn.LayerNorm(channels),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(channels, channels, bias=False),
                                        nn.LayerNorm(channels),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(channels, dim_proj, bias=False),
                                        nn.LayerNorm(dim_proj, elementwise_affine=False),
                                        )

        self.rnn_high = OPTS_RNN_CELLS[rnn_cell_high](dim_proj,dim_proj,dim_proj,name='linear')
        self.rnn_high.set_forward(forwardprop)

    # ============================================================
    # ============================================================

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), 
                        self.typeBN(planes * block.expansion))

        layers_ = [('0',block(self.inplanes, planes, stride, downsample, BN=self.typeBN))]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers_.append((str(i),block(self.inplanes, planes, BN=self.typeBN)))
      
        return nn.Sequential(col.OrderedDict(layers_))

    # ============================================================
    # ============================================================

    def forward(self, x, state, memory, grads):
        # Biological computation: Feedforward connection with delay

        new_state = []
        new_memory = []
        new_grads = []

        # 1st stage ----------------------------------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # B=1 x Ch x H x W

        # 2st stage: Dynamic in layers ---------------------------
        for idx_layer, layer in enumerate(self.layers_list):
            fforward_input = state[idx_layer-1] if idx_layer > 0 else x
            state_l, memory_l, grads_l = layer(fforward_input,state[idx_layer:],memory[idx_layer], grads[idx_layer])
            new_state.append(state_l)
            new_memory.append(memory_l)
            new_grads.append(grads_l)

        # 3st stage: Dynamic in Head -----------------------------

        if new_state[-1] is not None:
            r = self.avgpool(new_state[-1])
            r = flatten(r, 1)
            r = self.head(r)
        else:
            r = None 

        state_high, memory_high, grads_high = self.rnn_high(r,state[-1],memory[-1],grads[-1])

        new_state.append(state_high)
        new_memory.append(memory_high)
        new_grads.append(grads_high)

        return new_state, new_memory, new_grads

# -------------------------------------------------------------

class RNN_layer(nn.Module):
    def __init__(self,fforward_stage,feedbacks, rnn_cell, forwardprop):

        super(RNN_layer, self).__init__()

        # Operations for h(s_L-1,t) ---------------------------------------------------
        self.fforward_layer = fforward_stage
        planes = fforward_stage[0].conv1.in_channels

        # Feedbacks information (s_L+k, t-1) ------------------------------------------
        self.num_feedbacks    = len(feedbacks)
        self.fb_connections   = [cfg[0] for cfg in feedbacks]
        self.fb_in_ch         = [cfg[1] for cfg in feedbacks]
        
        # Aggregator (s_L-1,t and s_L+k,t-1) ------------------------------------------
        self.rsz_operator = nn.Linear(sum(self.fb_in_ch), planes, bias=False) if len(self.fb_in_ch)>0 else None
        self.activation = nn.ReLU()

        # RNN dynamic -----------------------------------------------------------------
        dim_rnn = fforward_stage[0].conv1.out_channels
        self.rnn_dynamic = OPTS_RNN_CELLS[rnn_cell](dim_rnn, None, None)
        self.rnn_dynamic.set_forward(forwardprop)

    def forward(self,fforward_state,prev_states,prev_memory, grads):

        if fforward_state is None:
            return None, None, None

        # Aggregate information of feedbacks ---------------------------------------
        Ba,Ch,H,W = fforward_state.shape

        feedback_component = [torch.zeros(Ba,x,H,W).cuda() for x in self.fb_in_ch] if self.num_feedbacks > 0 else []

        for i, connection in enumerate(self.fb_connections):
            distance = connection[0]-connection[1]
            if prev_states[distance] is not None:
                out_fback = self.activation(prev_states[distance])  # Batch x Channels_or x H_or x W_or
                out_fback = torch.nn.functional.interpolate(out_fback, size=fforward_state.size()[2:], mode='bilinear', align_corners=False)  # Batch x Channels_or x H_new x W_new
                feedback_component[i] = out_fback
        
        # ------------------------
 
        yy = [self.rsz_operator(torch.cat(feedback_component,dim=1).permute(0,2,3,1)).permute(0,3,1,2)] if len(feedback_component)>0 else []
        out = torch.stack(yy + [fforward_state], dim=-1)
        h = torch.sum(out, dim=-1)

        # ------------------------
        h = self.fforward_layer(h)

        # Dynamic of states/memory --------------------------------------------------
        #Check 
        internal_state = prev_states[0]
        internal_memory = prev_memory
        # internal_state  = prev_states[0] if prev_states[0] is not None else torch.zeros_like(h)
        # internal_memory = prev_memory if prev_memory is not None else torch.zeros_like(h)

        new_state, new_memory, new_grads = self.rnn_dynamic(h,internal_state,internal_memory, grads)
        return new_state, new_memory, new_grads