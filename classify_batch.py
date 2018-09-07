import matplotlib
matplotlib.use('Agg')
import glob
#import models.vgg16 as vgg16
import numpy as np
import tensorflow as tf
import argparse
import misc.utils as utils
#import models.vgg_utils as vgg_utils
import os
import time

def run(filename_list, class_label, out_dir):
    arch_path = '/mnt/md0/Projects/FathomNet/Training_Files/Midwater/snapshots/'
    epoch = 35
    nb_classes = 15
    mean_img_path = '/mnt/md0/Projects/FathomNet/Training_Files/Midwater/mean_img.png'
    activation_layer = 'activation_46'
    utils.grad_CAM_plus_batch(filename_list, class_label, arch_path, epoch, nb_classes, out_dir, mean_img_path, activation_layer='activation_46')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--class_label', default=-1, type=int, help='if -1 (default) choose predicted class, else user supplied int')
    parser.add_argument('-gpu', '--gpu_device', default="0", type=str, help='if 0 (default) choose gpu 0, else user supplied int')
    parser.add_argument('-d', '--dir_name', type=str, help="Specify image directory for Grad-CAM++ visualization")
    parser.add_argument('-o', '--out_dir', type=str, help="Output directory to write to")
    parser.add_argument('-b', '--batch_size', default=128, type=int, help="Number of images to process in a batch")
    args = parser.parse_args()
    gpu_id = args.gpu_device
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    filenames = glob.glob(os.path.join(args.dir_name,'*.png'))
    idx = 0
    while idx < len(filenames):
        start_time = time.time()
        if idx+args.batch_size > len(filenames)-1:
            filename_list = filenames[idx:-1]
        else:
            filename_list = filenames[idx:idx+args.batch_size]
        run(filename_list, args.class_label, args.out_dir)
        idx += args.batch_size
        print(f'Run time: {time.time() - start_time}')

if __name__ == '__main__':
    main()
