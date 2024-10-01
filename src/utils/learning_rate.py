import math

def adjust_learning_rate(optimizer, init_lr, epoch, num_epoch):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr if 'fix_lr' in param_group and param_group['fix_lr'] else cur_lr