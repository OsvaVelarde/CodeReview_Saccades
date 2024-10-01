import argparse
import time
import os
import numpy as np
import pickle

from dataset.dataset import SaccadesParraLab
from torch.utils.data import random_split, DataLoader

import utils.load_save as load_save
from utils.meters import ProgressMeter, AverageMeter

from torch import save, transpose, no_grad
from utils.learning_rate import adjust_learning_rate

folderdata = {'fixed': 'seq_fixed_len',
              'event': 'seq_event_cuts',
              'cont': 'seq_cont_cuts'}

# ======================================================================
# ======================================================================

parser = argparse.ArgumentParser(description='Recurrent Feedback for Saccades')

parser.add_argument('--PATH', required=True, help='Path to results')
parser.add_argument('--title', required=True, help='Name of experiment')
parser.add_argument('--datapath', required=True, help='Path to dataset')
parser.add_argument('--cfgfilename', help='Projector CFG')
parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
parser.add_argument('--gpu-number', type=int, default=0)

parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint')

parser.add_argument('-p', '--print-freq', default=10, type=int, help='Print frequency')

args = parser.parse_args()
cfgdict = load_save.read_cfgfile(args.cfgfilename)

# ======================================================================
# ========================== DATASET ===================================
datapath = args.datapath + folderdata[cfgdict['training']['cfg']['type_seq']] + '/pts/'
dataset = SaccadesParraLab(datapath)
num_samples = len(dataset)
num_train_samples = int(0.8 * num_samples)
num_val_samples = int(0 * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train_samples,num_val_samples, num_test_samples])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# ======================================================================
# ========================== NETWORK MODEL =============================
struct_cfg = {
	'projector_name':cfgdict['projector']['name'],
	'cfg_projector':cfgdict['projector']['cfg'],
	'pred_dim':cfgdict['predictor']['cfg']['dim_pred'],
	'similarity':'cosine'}

model_cfg = {
	'device_type': args.device_type,
	'gpu_number': args.gpu_number,
	'resume': args.resume,
	'cfg': struct_cfg}

model, device, initial_epoch =  load_save.initialize_model(model_cfg)

simclr_files = {'backbone_file': args.PATH + 'models/Backbone_Max_Lukas.pth',
                'head_file': args.PATH + 'models/Head_Max_Lukas.pth'}

#model.module.encoder = load_save.upload_resnet_simclr(model.module.encoder,device, **simclr_files)
model.module.encoder = load_save.upload_recresnet_simclr(model.module.encoder,device, **simclr_files)

optimizer_parameters = {
    'cfg': cfgdict['optimizer']['cfg'],
    'fix_lr_projector':False,
    'fix_lr_predictor':cfgdict['predictor']['cfg']['fix_lr'],
    'optimizer_path': None}

optimizer = load_save.initialize_optimizer(model,optimizer_parameters)
init_lr = cfgdict['optimizer']['cfg']['lr']

# ======================================================================
# ============================ OUTPUTS =================================
# Checkpoints and results folder ---------------------------
checkpoint_folder = args.PATH + 'results/checkpoints/'
evolution_folder = args.PATH +  'results/training/'
recurrence_folder = args.PATH +  'results/loss_recurrence/' + args.title + '/'

if not os.path.isdir(checkpoint_folder): os.mkdir(checkpoint_folder)
if not os.path.isdir(evolution_folder): os.mkdir(evolution_folder)
if not os.path.isdir(recurrence_folder): os.makedirs(recurrence_folder)

# ======================================================================
# ========================== TRAINING ==================================
num_epochs = cfgdict['training']['cfg']['epochs']
interm_epoch = initial_epoch + int(num_epochs/2)
last_epoch = initial_epoch + num_epochs

train_loss_history = []
eval_loss_history = []

# Training process -----------------------------------------
for epoch in range(initial_epoch, last_epoch):
 
    # Meters -----------------------------------------------
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))

    # Online Evaluation ------------------------------------
    model.eval()
    
    with no_grad():
        test_loss = 0.0
        factor = 0.0
        loss_vs_sacc = {}

        for batch_idx, batch in enumerate(test_loader):

            name_seq, seq_patches = batch
            seq_patches = transpose(seq_patches,0,1)
            seq_patches = seq_patches.to(device)

            if len(seq_patches)<model.module.encoder.num_layers + 1:
                continue

            ss_predictions, ss_seq_loss, ss_loss = model(seq_patches)
            loss_vs_sacc[name_seq] = ss_seq_loss
            test_loss += ss_loss.item()

            factor += 1.0 

    # ------------------------------------------------------
    with open(recurrence_folder + 'test_epoch_' + str(epoch) + '.pkl', 'wb') as f:
        pickle.dump(loss_vs_sacc, f)

    eval_loss_epoch = test_loss/factor
    eval_loss_history.append(eval_loss_epoch)
    print('Epoch:',epoch,'Loss Eval:',eval_loss_epoch)

    # Training ---------------------------------------------
    model.train()
    train_loss = 0.0
    factor = 0.0
    train_loss_vs_sacc = {}
    end = time.time()

    for batch_idx, batch in enumerate(train_loader):

        data_time.update(time.time() - end)
        optimizer.zero_grad()

        name_seq, seq_patches = batch
        seq_patches = transpose(seq_patches,0,1)
        seq_patches = seq_patches.to(device)

        if len(seq_patches)<model.module.encoder.num_layers + 1:
            continue

        ss_predictions, ss_seq_loss, ss_loss = model(seq_patches)
        train_loss_vs_sacc[name_seq] = ss_seq_loss
        train_loss += ss_loss.item()

        losses.update(ss_loss.item())
        ss_loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        factor += 1.0

        if batch_idx % args.print_freq == 0: progress.display(batch_idx)

    with open(recurrence_folder + 'train_epoch_' + str(epoch) + '.pkl', 'wb') as f:
        pickle.dump(train_loss_vs_sacc, f)
    
    train_loss_epoch = train_loss/factor
    train_loss_history.append(train_loss_epoch)
    print('Epoch:',epoch,'Loss Train:',train_loss_epoch)

    # Save checkpoint --------------------------------------
    print('Checkpoint -- ')
    state = {
        'model': model.state_dict(),
        'loss': eval_loss_epoch,
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch+1}
    
    save(state, checkpoint_folder +'ckpt_'+ args.title + '.pth')
    num_stag = 0

    adjust_learning_rate(optimizer, init_lr, epoch-initial_epoch, num_epochs)

# Save results
np.savetxt(evolution_folder + args.title +'.dat', np.array([train_loss_history, eval_loss_history]), fmt="%.5f")