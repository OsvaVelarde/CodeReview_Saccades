import argparse
import os

from dataset.dataset import SaccadesParraLab
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils.load_save as load_save
from torch import transpose, no_grad
from torch.nn import Identity
from models.rnn_cells import non_dynamic_cell
from numpy.random import permutation

import pickle

folderdata = {'fixed': 'seq_fixed_len',
              'event': 'seq_event_cuts',
              'cont': 'seq_cont_cuts',
              'chapter': 'seq_chapters'}

# ======================================================================
# ======================================================================

parser = argparse.ArgumentParser(description='Recurrent Feedback for Saccades')

parser.add_argument('--PATH', required=True, help='Path to results')
parser.add_argument('--title', required=True, help='Name of experiment')
parser.add_argument('--datapath', required=True, help='Path to dataset')
parser.add_argument('--type-seq', required=True, help='Type of sequence')
parser.add_argument('--cfgfilename', help='Projector CFG')
parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
parser.add_argument('--gpu-number', type=int, default=0)
parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint')
parser.add_argument('--remove-rec', action='store_true')

args = parser.parse_args()
cfgdict = load_save.read_cfgfile(args.cfgfilename)

# ======================================================================
# ========================== DATASET ===================================
datapath = args.datapath + folderdata[args.type_seq] + '/pts/'
dataset = SaccadesParraLab(datapath)

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

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

if args.remove_rec:
    print('Running without recurrence')
    model.module.predictor = Identity()
    model.module.encoder.rnn_high = non_dynamic_cell(None,None,None)

# =====================================================================
# =========================== OUTPUTS =================================

predictions_folder = args.PATH + 'results/predictions/' 
if not os.path.isdir(predictions_folder): os.makedirs(predictions_folder)

# =====================================================================
# ======================== EVALUATION =================================

model.eval()
test_loss = 0.0
factor = 0.0
predictions = {}
loss_vs_sacc = {}

with no_grad():
    for batch_idx, batch in enumerate(loader):
        name_seq, seq_patches = batch
        seq_patches = transpose(seq_patches,0,1)
        seq_patches = seq_patches.to(device)

        if len(seq_patches)<model.module.encoder.num_layers + 1:
            continue

        ss_predictions, ss_seq_loss, ss_loss = model(seq_patches)
        predictions[name_seq] = ss_predictions
        loss_vs_sacc[name_seq] = ss_seq_loss
        test_loss += ss_loss.item()

        factor += 1.0 

with open(predictions_folder + args.title + '.pkl', 'wb') as f:
    pickle.dump(predictions, f)

with open(predictions_folder + args.title + '_loss.pkl', 'wb') as f:
    pickle.dump(loss_vs_sacc, f)

eval_loss_epoch = test_loss/factor
print('Loss Eval:',eval_loss_epoch)