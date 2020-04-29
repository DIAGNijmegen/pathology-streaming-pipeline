import wandb
import subprocess

def initialize_wandb(project_name, exp_name, nets, key=''):
    command = "wandb login " + key
    subprocess.call(command, shell=True)
    wandb.init(project=project_name, name=exp_name, resume=exp_name)
    for net in nets: wandb.watch(net, log='all')

def write_log(name, tensor, epoch, x_name='epoch'):
    try:
        wandb.log({x_name: epoch, name: tensor})
    except Exception:
        pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_checkpoint(evaluator, dataset_name, avg_acc, avg_loss, epoch=0):
    write_log('loss_' + dataset_name, avg_loss, epoch)
    write_log('acc_' + dataset_name, avg_acc, epoch)
    print(f'Checkpoint {epoch}, {dataset_name} acc: {avg_acc:.4f}, {dataset_name} loss: {avg_loss:.4f}')

