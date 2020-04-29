import torch
from torch.nn.parameter import Parameter
# import apex.amp as amp  # type:ignore
import torchvision

from scnn import StreamingCNN, StreamingConv2d

if '1.6' in torch.__version__:  # type:ignore
    from torch.cuda.amp import GradScaler
    from torch.cuda.amp import autocast

def test_net(stream_net, image, target, tile_size=256, convert=(), mixedprecision=False, verbose=True):
    # criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(stream_net.parameters(), lr=1e-5)

    sCNN = StreamingCNN(stream_net,
                        tile_shape=(1, 3, tile_size, tile_size),
                        verbose=False,
                        normalize_on_gpu=False,
                        copy_to_gpu=True)

    if mixedprecision:
        sCNN.dtype = torch.half
        scaler = GradScaler(init_scale=1.0)

    stream_net.zero_grad()
    sCNN.disable()
    sCNN.enable()
    state_dict = sCNN.state_dict()
    sCNN.load_state_dict(state_dict)

    if mixedprecision:
        with autocast():
            with torch.no_grad():
                str_output = sCNN.forward(image)
            str_output.requires_grad = True
            loss = torch.sum(str_output.float()) / 1e4
        scaler.scale(loss).backward()  # type:ignore
        sCNN.backward(image, str_output.grad)
    else:
        str_output = sCNN.forward(image)
        str_output.requires_grad = True
        loss = torch.sum(str_output.float()) / 1e4
        loss.backward()
        sCNN.backward(image, str_output.grad)

    streaming_conv_gradients = []
    for i, layer in enumerate(stream_net.modules()):
        if isinstance(layer, StreamingConv2d):
            if layer.weight.grad is not None:
                streaming_conv_gradients.append(layer.weight.grad.clone()) 

    sCNN.disable()
    stream_net.zero_grad()
    optimizer.zero_grad()

    # normal
    if mixedprecision:
        with autocast(enabled=True): 
            str_output = stream_net(image[None])
            loss = torch.sum(str_output) / 1e4
            print(loss)

        scaler.scale(loss).backward()  # type:ignore
    else:
        str_output = stream_net(image[None])
        loss = torch.sum(str_output.float()) / 1e4
        loss.backward()

    normal_conv_gradients = []
    j = 0
    for i, layer in enumerate(stream_net.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            if layer.weight.grad is not None:
                normal_conv_gradients.append(layer.weight.grad) 
                j += 1

    passed = True
    assert len(streaming_conv_gradients) > 0
    for i in range(len(streaming_conv_gradients)):
        max_diff = torch.max(torch.abs(streaming_conv_gradients[i].data - normal_conv_gradients[i].data))  # type:ignore
        if verbose:
            print("Conv layer", i, "\t max diff between gradients:", float(max_diff))
            print('Max str', torch.max(streaming_conv_gradients[i].data).item(), 'max norm', torch.max(normal_conv_gradients[i].data).item())
        if not mixedprecision and max_diff > 1e-12: passed = False
        elif mixedprecision and max_diff > 1e-2: passed = False
    return passed

if __name__ == '__main__':
    img_size = 512

    image = torch.FloatTensor(3, 1024, 1024).normal_(0, 1)  # type:ignore
    target = torch.tensor(1).fill_(1e5)  # type:ignore
    image = image.cuda().double()
    target = target.cuda().double()

    print("Testing resnet-34")
    net = torchvision.models.resnet34(pretrained=True).double().cuda()
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.bias = Parameter(torch.zeros(m.out_channels, dtype=torch.double).cuda())  # type:ignore
            m.register_parameter('bias', m.bias)  # type:ignore | may not be needed, but does not harm
    stream_net = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
    for l in stream_net.modules():
        if isinstance(l, torch.nn.BatchNorm2d): l.eval()
    passed = test_net(stream_net, image, target, tile_size=512)
    if passed: print("resnet-34 tests passed")
    else: print("!! resnet-34 tests did NOT pass")

    print("")

    print("Testing mobilenetv2")
    from models.mobilenetv2 import mobilenet_v2
    net = mobilenet_v2().double().cuda()
    stream_net = net.features[0:4]
    for l in stream_net.modules():
        if isinstance(l, torch.nn.BatchNorm2d): l.eval()
    passed = test_net(stream_net, image, target, convert=(torch.nn.ReLU6, torch.nn.ReLU))
    if passed: print("mobilenetv2 tests passed")
    else: print("!! mobilenetv2 tests did NOT pass")

    print("")
    
    print("Testing uneven (400 x 800) image-size mobilenetv2")
    image = torch.FloatTensor(3, 400, 800).normal_(0, 1).cuda().double()  # type:ignore
    passed = test_net(stream_net, image, target, convert=(torch.nn.ReLU6, torch.nn.ReLU))
    if passed: print("mobilenetv2 uneven (400 x 800) image-size tests passed")
    else: print("!! mobilenetv2 uneven (400 x 800) image-size tests NOT pass")

    print("")

    print("Testing uneven (800 x 400) image-size mobilenetv2")
    image = torch.FloatTensor(3, 800, 400).normal_(0, 1).cuda().double()  # type:ignore
    passed = test_net(stream_net, image, target, convert=(torch.nn.ReLU6, torch.nn.ReLU))
    if passed: print("mobilenetv2 uneven (800 x 400) image-size tests passed")
    else: print("!! mobilenetv2 uneven (800 x 400) image-size tests did NOT pass")

    print("")

    print("Testing small (256 x 256) image-size mobilenetv2")
    image = torch.FloatTensor(3, 256, 256).normal_(0, 1).cuda().double()  # type:ignore
    passed = test_net(stream_net, image, target, convert=(torch.nn.ReLU6, torch.nn.ReLU))
    if passed: print("mobilenetv2 small (256 x 256) image-size tests passed")
    else: print("!! mobilenetv2 small (256 x 256) image-size tests did NOT pass")

    print("")

    print("Testing irregular (256 x 800) image-size mobilenetv2")
    image = torch.FloatTensor(3, 256, 800).normal_(0, 1).cuda().double()  # type:ignore
    passed = test_net(stream_net, image, target, convert=(torch.nn.ReLU6, torch.nn.ReLU))
    if passed: print("mobilenetv2 irregular (256 x 800) image-size tests passed")
    else: print("!! mobilenetv2 irregular (256 x 800) image-size tests did NOT pass")

    print("")

    print("Testing irregular (800 x 256) image-size mobilenetv2")
    image = torch.FloatTensor(3, 800, 256).normal_(0, 1).cuda().double()  # type:ignore
    passed = test_net(stream_net, image, target, convert=(torch.nn.ReLU6, torch.nn.ReLU))
    if passed: print("mobilenetv2 irregular (800 x 256) image-size tests passed")
    else: print("!! mobilenetv2 irregular (800 x 256) image-size tests did NOT pass")

    print("")

    print("Testing very small (224 x 256) image-size mobilenetv2")
    image = torch.FloatTensor(3, 224, 256).normal_(0, 1).cuda().double()  # type:ignore
    passed = test_net(stream_net, image, target, convert=(torch.nn.ReLU6, torch.nn.ReLU))
    if passed: print("mobilenetv2 very small (128 x 256) image-size tests passed")
    else: print("!! mobilenetv2 very small (800 x 256) image-size tests did NOT pass")

    if '1.6' in torch.__version__:  # type:ignore
        print("")
        print("Testing mixed precision resnet-34")
        image = torch.FloatTensor(3, 1024, 1024).normal_(0, 1).cuda()  # type:ignore
        image = image.cuda().float()
        target = target.cuda().float()
        net = torchvision.models.resnet34(pretrained=True).cuda()
        stream_net = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        for l in stream_net.modules():
            if isinstance(l, torch.nn.BatchNorm2d): 
                l.eval()
        passed = test_net(stream_net, image, target, mixedprecision=True, verbose=True)
        if passed: print("mixed precision resnet-34 tests passed")
        else: print("!! mixed precision resnet-34 tests did NOT pass")
        print("")
    else:
        print("Skipping mixed precision test, older pytorch version")
