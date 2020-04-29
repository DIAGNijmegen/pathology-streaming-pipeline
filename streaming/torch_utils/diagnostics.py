# from itertools import chain
import torch
import torch.distributed as dist
# from torch_utils.utils import write_log

def log_params_norm(parameters, step):
    return
# for name, param in parameters:
# if 'weight' in name:
# write_log(name, float(param.grad.norm() / param.data.norm()), step)
# write_log(name, float(param.grad.mean()), step, x_name='batch')
# write_log(name + '_mean', float(param.grad.mean() / param.data.mean()), step)
# write_log(name + '_std', float(param.grad.std() / param.data.std()), step)

def check_params_distributed(net, n_gpus, rank):
    param = next(net.parameters())
    tensor_list = [param.new_empty(param.shape) for i in range(n_gpus)]
    dist.all_gather(tensor_list, param)
    if rank == 0:
        for i in range(n_gpus):
            if not torch.isnan(tensor_list[0]).any() and \
                    not torch.isnan(tensor_list[1]).any() and \
                    not torch.allclose(tensor_list[0], tensor_list[i]):
                print('WARNING!!!! GRADS NOT EQUAL')
                # from pdb import set_trace; set_trace()

    if param.grad is not None:
        tensor_list = [param.new_empty(param.shape) for i in range(n_gpus)]
        dist.all_gather(tensor_list, param.grad)
        if rank == 0:
            for i in range(n_gpus):
                if not torch.isnan(tensor_list[0]).any() and \
                        not torch.isnan(tensor_list[1]).any() and \
                        not torch.allclose(tensor_list[0], tensor_list[i]):
                    print('WARNING!!!! GRADS NOT EQUAL')
                    # from pdb import set_trace; set_trace()
