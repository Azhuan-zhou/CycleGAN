from options.test_options import TestOptions
from models import create_model
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from util import util
# 定义图像转换操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为 Tensor
])


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


def init_model(opt):
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)
    #path = './checkpoints/{}/latest_net_G.pth'.format(opt.name)
    #checkpoint = torch.load(path)
    #model.netG.load_state_dict(checkpoint)
    return model


def generate(model, data):
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image Azhuan
    return visuals


if __name__ == '__main__':
    A_paths = ['./Azhuan/A/4.png']
    B_paths = ['./Azhuan/B/4.png']
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test --model test
    opt.no_dropout = True
    opt.name = 'style_vangogh_pretrained'
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.no_flip = True  # no flip; comment this line if Azhuan on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the Azhuan to a HTML file.
    data = prepare_data(A_paths)
    model = init_model(opt)
    results = generate(model,data)['fake']
    l = len(A_paths)
    for i in range(l):
        image = results[i].unsqueeze(0)
        image = util.tensor2im(image)
        util.save_image(image,B_paths[i],opt.aspect_ratio)