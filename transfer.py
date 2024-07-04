from options.test_options import TestOptions
from models import create_model
from PIL import Image
import torchvision.transforms as transforms
import torch
import argparse
import os
from util import util
# 定义图像转换操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为 Tensor
])


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

def load_networks(model,path):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    print('loading the model from %s' % path)
    net = model.netG
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(path, map_location=str(model.device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    net.load_state_dict(state_dict)

def prepare_data(pathB):
    data = {}
    imageB = []
    for path in pathB:
        image = Image.open(path)
        image = image.convert('RGB')
        image = transform(image)
        # 添加一个批次维度
        image = image.unsqueeze(0)  # 形状变为 (1, C, H, W)
        imageB.append(image)
    imageB = torch.cat(imageB, dim=0)
    data['A'] = imageB
    data['A_paths'] = pathB
    return data


def init_model(opt,path):
    model = create_model(opt)  # create a model given opt.model and other options
    load_networks(model,path)

    return model


def generate(model, data):
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image Azhuan
    return visuals


if __name__ == '__main__':

    A_paths = os.listdir('./Azhuan/A/')
    B_paths = ['./Azhuan/B/{}'.format(path) for path in A_paths]
    A_paths = ['./Azhuan/A/{}'.format(path) for path in A_paths]
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test --model test
    path = './checkpoints/{}/{}'.format(opt.model_name, opt.checkpoint_name)
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.no_flip = True  # no flip; comment this line if Azhuan on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the Azhuan to a HTML file.
    data = prepare_data(A_paths)
    model = init_model(opt,path)
    results = generate(model,data)['fake']
    l = len(A_paths)
    for i in range(l):
        image = results[i].unsqueeze(0)
        image = util.tensor2im(image)
        util.save_image(image,B_paths[i],opt.aspect_ratio)
