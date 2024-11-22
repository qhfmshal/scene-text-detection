import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

import numpy as np
import random

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--train_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--val_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--pretrained', type=str, default = 'train')


    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(train_data_dir, val_data_dir, model_dir, pretrained, device, num_workers, image_size, input_size,  batch_size,
                learning_rate, max_epoch, save_interval):

    seed_everything(12)
    train_dataset = SceneTextDataset(train_data_dir, split='train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# =================================================================================
# val
    val_dataset = SceneTextDataset(val_data_dir, split='validation', image_size=image_size, crop_size=input_size)
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    best_val_loss = np.inf
# =================================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()

    if pretrained == 'pretrained' :
      weights_path = os.path.join(model_dir,'best.pth')
      model.load_state_dict(torch.load(weights_path))
      
    elif pretrained == 'resume' :
      weights_path = os.path.join(model_dir,'latest.pth')
      model.load_state_dict(torch.load(weights_path))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        print(f'Epoch{epoch+1}')
        train_epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train = loss.item()
                train_epoch_loss += loss_train

                pbar.update(1)
                Train_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(Train_dict)
                
        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            train_epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))
# =================================================================================
# val
        with torch.no_grad():
          model.eval()
          val_epoch_loss = 0
          print('Validation Start')
          with tqdm(total=val_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
              loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
              val_epoch_loss += loss.item()
              pbar.update(1)
              
            mean_val_loss = val_epoch_loss / val_num_batches
            if best_val_loss > mean_val_loss:
              best_val_loss = mean_val_loss
              best_val_loss_epoch = epoch+1
              ckpt_path = osp.join(model_dir, 'best.pth')
              torch.save(model.state_dict(), ckpt_path)
# =================================================================================
        print('Best loss: {:.4f} | Epoch: {}'.format(best_val_loss,best_val_loss_epoch))
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_path = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_path)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
