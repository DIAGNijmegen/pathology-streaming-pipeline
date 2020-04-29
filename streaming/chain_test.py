import torch
from torch.nn.parameter import Parameter
import torchvision

from torch_utils.scnn import StreamingCNN, StreamingConv2d

# if '1.6' in torch.__version__:
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

torch.backends.cudnn.benchmark = True  # type:ignore

image = torch.zeros((3, 16384*2, 16384*2), dtype=torch.uint8)
target = torch.tensor(1).fill_(1e5)

mixedprecision = False

net = torchvision.models.resnet34(pretrained=True).cuda()

for l in net.modules():
    if isinstance(l, torch.nn.BatchNorm2d): l.eval()
        
for m in net.modules():
    if isinstance(m, torch.nn.Conv2d):
        m.bias = Parameter(torch.zeros(m.out_channels, dtype=torch.float).cuda())  # type:ignore
        m.register_parameter('bias', m.bias)  # type:ignore | may not be needed, but does not harm

stream_net_1 = torch.nn.Sequential(net.conv1,
                                   net.bn1,
                                   net.relu,
                                   net.maxpool,
                                   net.layer1,
                                   net.layer2,
                                   net.layer3,
                                   net.layer4[0])

net.layer4 = net.layer4[1:]
net.layer1 = torch.nn.Sequential()
net.layer2 = torch.nn.Sequential()
net.layer3 = torch.nn.Sequential()
net.conv1 = torch.nn.Sequential()
net.bn1 = torch.nn.Sequential()
net.relu = torch.nn.Sequential()
net.maxpool = torch.nn.Sequential()
net.avgpool = torch.nn.AdaptiveMaxPool2d(1)

state_dict = torch.load('test_state_dict')
sCNN = StreamingCNN(stream_net_1, tile_shape=(1, 3, 3296, 3296), verbose=True, copy_to_gpu=False, statistics_on_cpu=False, state_dict=state_dict)

output = torch.ones((1, 512, 1024, 1024)).cuda()
net(output[0])

if mixedprecision:
    sCNN.dtype = torch.half

scaler = GradScaler(init_scale=1.0)

with autocast():
    with torch.no_grad():
        str_output = sCNN.forward(image)

print(str_output.shape)
str_output = str_output.cuda()
str_output.requires_grad = True
output = net(str_output)
print(output)

loss = torch.sum(str_output.float()) / 1e5

scaler.scale(loss).backward()

grad = str_output.grad.cpu().half()
del str_output, loss, output
from pdb import set_trace; set_trace()

torch.cuda.memory_allocated()

sCNN.backward(image, grad)

# sCNN_2 = StreamingCNN(stream_net_2, tile_shape=(1, 128, 512, 512), verbose=True, normalize_on_gpu=False)


# str_output_1 = sCNN.forward(image)


print(output)
# str_output_2 = sCNN_2.forward(str_output_1[0])
