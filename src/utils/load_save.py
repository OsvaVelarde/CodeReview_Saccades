import configparser
import ast

import torch
from torch.nn import DataParallel, Parameter
import torch.backends.cudnn as cudnn

from models.rjepa import RJEPA

# ---------------------------------------------------------------------
BOOL_MAP = {'False':False, 'True':True}
CONV_MAP = {'int': int, 'float': float, 'list':ast.literal_eval, 'bool':BOOL_MAP.get, 'str':str}
SUFIX_MAP = {'int': 4, 'float': 6, 'list':5, 'bool':5, 'str':4}

def read_cfgfile(filename):
    cfg = configparser.ConfigParser()
    cfg.read(filename)
    data = {}

    for ss in cfg.sections():
    	section, name = ss.split(":")
    	data[section]={'name':name, 'cfg':{}}

    	for key, value in cfg.items(ss):
    		sufix = key.split('_')[-1]

    		conversion = CONV_MAP[sufix]
    		value_c = conversion(value)
    		key_c = key[:-SUFIX_MAP[sufix]]

    		data[section]['cfg'][key_c] = value_c

    return data
# ---------------------------------------------------------------------

def initialize_model(parameters):

	model = RJEPA(**parameters["cfg"])

	if parameters["device_type"] == 'gpu':
		model = DataParallel(model)
		cudnn.benchmark = True

	epoch = 0
	
	if parameters['resume']:
		checkpoint = torch.load(parameters["resume"])
		model.load_state_dict(checkpoint["model"],strict=False)
		epoch = checkpoint['epoch']

	if (parameters["device_type"] == "gpu") and torch.has_cudnn:
		device = torch.device("cuda:{}".format(parameters["gpu_number"]))
	else:
		device = torch.device("cpu")

	model = model.to(device)
	return model, device, epoch
# ---------------------------------------------------------------------

def initialize_optimizer(model, parameters):
	cfg = parameters["cfg"]

	optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': parameters['fix_lr_projector']},
					{'params': model.module.predictor.parameters(), 'fix_lr': parameters['fix_lr_predictor']}]

	optimizer = torch.optim.Adam(optim_params, **cfg)

	if parameters["optimizer_path"]:
		checkpoint = torch.load(parameters["optimizer_path"])
		optimizer.load_state_dict(checkpoint['optimizer'])

	return optimizer

# -----------------------------------------------------------------
# -----------------------------------------------------------------

def upload_resnet_simclr(model,device,backbone_file,head_file):
    b_parameters = torch.load(backbone_file)['state_dict']
    h_parameters = torch.load(head_file)['state_dict']

    # ----------------------------------------------------------------------------------
    model.conv1.weight = Parameter(b_parameters['conv1.weight'].to(device),requires_grad=False)
    model.bn1.weight = Parameter(b_parameters['bn1.weight'].to(device),requires_grad=False)
    model.bn1.bias = Parameter(b_parameters['bn1.bias'].to(device),requires_grad=False)
    model.bn1.running_mean = Parameter(b_parameters['bn1.running_mean'].to(device),requires_grad=False)
    model.bn1.running_var = Parameter(b_parameters['bn1.running_var'].to(device),requires_grad=False)
    model.bn1.momentum = 0.9

    for ll, layer in enumerate(model.layers_list):
    	for bb, block in enumerate(layer):
    		block.bn1.running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_mean'].to(device),requires_grad=False)
    		block.bn1.running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_var'].to(device),requires_grad=False)
    		block.bn2.running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_mean'].to(device),requires_grad=False)
    		block.bn2.running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_var'].to(device),requires_grad=False)
    		block.bn3.running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.running_mean'].to(device),requires_grad=False)
    		block.bn3.running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.running_var'].to(device),requires_grad=False)

    		block.bn1.momentum = 0.9
    		block.bn2.momentum = 0.9
    		block.bn3.momentum = 0.9

    		block.conv1.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv1.weight'].to(device),requires_grad=False)
    		block.bn1.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.weight'].to(device),requires_grad=False)
    		block.bn1.bias = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.bias'].to(device),requires_grad=False)
    		block.conv2.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv2.weight'].to(device),requires_grad=False)
    		block.bn2.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.weight'].to(device),requires_grad=False)
    		block.bn2.bias = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.bias'].to(device),requires_grad=False)
    		block.conv3.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv3.weight'].to(device),requires_grad=False)
    		block.bn3.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.weight'].to(device),requires_grad=False)
    		block.bn3.bias = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.bias'].to(device),requires_grad=False)

    	layer[0].downsample[0].weight = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.0.weight'].to(device),requires_grad=False)
    	layer[0].downsample[1].weight = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.weight'].to(device),requires_grad=False)
    	layer[0].downsample[1].bias = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.bias'].to(device),requires_grad=False)

    	layer[0].downsample[1].running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.running_mean'].to(device),requires_grad=False)
    	layer[0].downsample[1].running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.running_var'].to(device),requires_grad=False)

    # ----------------------------------------------------------------------------------

    # for ii in [0,1,2]:
    # 	model.head[2*ii].weight = Parameter(h_parameters[str(3*ii)+'.weight'].T.to(device),requires_grad=False)

    for ii in [0,3,6]:
    	model.head[ii].weight = Parameter(h_parameters[str(ii)+'.weight'].T.to(device),requires_grad=False)

    for ii in [1,4,7]:
    	model.head[ii].weight = Parameter(h_parameters[str(ii)+'.weight'].to(device),requires_grad=False)
    	model.head[ii].bias = Parameter(h_parameters[str(ii)+'.bias'].to(device),requires_grad=False)
    	model.head[ii].running_mean = Parameter(h_parameters[str(ii)+'.running_mean'].to(device),requires_grad=False)
    	model.head[ii].running_var = Parameter(h_parameters[str(ii)+'.running_var'].to(device),requires_grad=False)

    return model


def upload_recresnet_simclr(model,device,backbone_file,head_file):
    b_parameters = torch.load(backbone_file)['state_dict']
    h_parameters = torch.load(head_file)['state_dict']

    # ----------------------------------------------------------------------------------
    model.conv1.weight = Parameter(b_parameters['conv1.weight'].to(device),requires_grad=False)
    model.bn1.weight = Parameter(b_parameters['bn1.weight'].to(device),requires_grad=False)
    model.bn1.bias = Parameter(b_parameters['bn1.bias'].to(device),requires_grad=False)
    model.bn1.running_mean = Parameter(b_parameters['bn1.running_mean'].to(device),requires_grad=False)
    model.bn1.running_var = Parameter(b_parameters['bn1.running_var'].to(device),requires_grad=False)
    model.bn1.momentum = 0.9

    for ll, layer in enumerate(model.layers_list):
    	for bb, block in enumerate(layer.fforward_layer):
    		block.bn1.running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_mean'].to(device),requires_grad=False)
    		block.bn1.running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.running_var'].to(device),requires_grad=False)
    		block.bn2.running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_mean'].to(device),requires_grad=False)
    		block.bn2.running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.running_var'].to(device),requires_grad=False)
    		block.bn3.running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.running_mean'].to(device),requires_grad=False)
    		block.bn3.running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.running_var'].to(device),requires_grad=False)

    		block.bn1.momentum = 0.9
    		block.bn2.momentum = 0.9
    		block.bn3.momentum = 0.9

    		block.conv1.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv1.weight'].to(device),requires_grad=False)
    		block.bn1.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.weight'].to(device),requires_grad=False)
    		block.bn1.bias = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn1.bias'].to(device),requires_grad=False)
    		block.conv2.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv2.weight'].to(device),requires_grad=False)
    		block.bn2.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.weight'].to(device),requires_grad=False)
    		block.bn2.bias = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn2.bias'].to(device),requires_grad=False)
    		block.conv3.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.conv3.weight'].to(device),requires_grad=False)
    		block.bn3.weight = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.weight'].to(device),requires_grad=False)
    		block.bn3.bias = Parameter(b_parameters['layer'+ str(ll+1) + '.' + str(bb) + '.bn3.bias'].to(device),requires_grad=False)

    	layer.fforward_layer[0].downsample[0].weight = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.0.weight'].to(device),requires_grad=False)
    	layer.fforward_layer[0].downsample[1].weight = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.weight'].to(device),requires_grad=False)
    	layer.fforward_layer[0].downsample[1].bias = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.bias'].to(device),requires_grad=False)

    	layer.fforward_layer[0].downsample[1].running_mean = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.running_mean'].to(device),requires_grad=False)
    	layer.fforward_layer[0].downsample[1].running_var = Parameter(b_parameters['layer'+ str(ll+1) + '.0.downsample.1.running_var'].to(device),requires_grad=False)

    # ----------------------------------------------------------------------------------

    # for ii in [0,1,2]:
    # 	model.head[2*ii].weight = Parameter(h_parameters[str(3*ii)+'.weight'].T.to(device),requires_grad=False)

    for ii in [0,3,6]:
    	model.head[ii].weight = Parameter(h_parameters[str(ii)+'.weight'].T.to(device),requires_grad=False)

    for ii in [1,4,7]:
    	model.head[ii].weight = Parameter(h_parameters[str(ii)+'.weight'].to(device),requires_grad=False)
    	model.head[ii].bias = Parameter(h_parameters[str(ii)+'.bias'].to(device),requires_grad=False)
    	model.head[ii].running_mean = Parameter(h_parameters[str(ii)+'.running_mean'].to(device),requires_grad=False)
    	model.head[ii].running_var = Parameter(h_parameters[str(ii)+'.running_var'].to(device),requires_grad=False)

    return model
