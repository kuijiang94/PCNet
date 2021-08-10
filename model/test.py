"""
## Rain-free and Residue Hand-in-Hand: A Progressive Coupled Network for Real-Time Image Deraining
## Kui Jiang, Zhongyuan Wang, Peng Yi, Chen Chen, Zheng Wang, Xiao Wang, Junjun Jiang, and Chia-Wen Lin
## https://www.researchgate.net/profile/Kui-Jiang-3/publication/351868487_PCNET_PROGRESSIVE_COUPLED_NETWORK_FOR_REAL-TIME_IMAGE_DERAINING/links/60adf917299bf13438e85cd0/PCNET-PROGRESSIVE-COUPLED-NETWORK-FOR-REAL-TIME-IMAGE-DERAINING.pdf
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
import time

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from thop import profile

from data_RGB import get_test_data
from PCNet1 import PCNet#STRN_woALL

from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using MSGN')

#parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')#results,lowlight\LOL1000,dehazing
parser.add_argument('--weights', default='./checkpoints/Deraining/models/PCNet/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')#STRN_woALL

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #args.gpus

model_restoration = WD2N()#,MSGN,STRN,MPRNet,WD2N

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
#model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200']
#datasets = ['Test1200']
for dataset in datasets:
    rgb_dir_test = os.path.join(args.result_dir, dataset, 'input')#input
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    #test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True) ### for windows testing
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True) ### for Linux testing
	
    result_dir  = os.path.join(args.result_dir, dataset, 'PCNet_best')
    if not os.path.exists(result_dir):
        utils.mkdir(result_dir)
    all_time =0
    count = 0
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            input_    = data_test[0].cuda()

            filenames = data_test[1]
            st_time=time.time()
            restored = model_restoration(input_)
            ed_time=time.time()
            cost_time=ed_time-st_time
            all_time +=cost_time
            count +=1
            #print('spent {} s.'.format(cost_time))
            #print(filenames)
            restored = torch.clamp(restored,0,1)
            #restored = torch.clamp(restored[0],0,1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
    print('spent {} s.'.format(all_time))
    print('spent {} s per item.'.format(all_time/(count)))#(ii+1)
flops, params = profile(model_restoration, (input_,))
print('flops: ', flops, 'params: ', params)