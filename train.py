import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

import cv2
import imageio

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator

from detection.transform import GeneralizedRCNNTransform

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def collate_function(data):
    return tuple(zip(*data))

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print("Checkpoint saved at", filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    angle_step_size = 20
    
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    angle_step_size = dataset_config['angle_step_size']
    prediction_method = train_config['angle_prediction_method']

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])
    
    st = SceneTextDataset('', root_dir=dataset_config['root_dir'],angle_step_size = angle_step_size,prediction_method = prediction_method,transform=transform)

    train_dataset = DataLoader(st,
                               batch_size=1,
                               shuffle=True,
                               num_workers=8,
                               collate_fn=collate_function
                               )
    
    # val_dataset = DataLoader(val,batch_size=1,shuffle=True,num_workers=6,collate_fn=collate_function)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                        min_size=600,
                                                        max_size=1000,
                                                        angle_step_size = angle_step_size,
                                                        prediction_method = prediction_method,
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'])

    faster_rcnn_model.train()
    print('Model Loaded...')
    faster_rcnn_model.to(device)
    print('Model moved to device...')
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=1E-4,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5, momentum=0.9)

    num_epochs = train_config['num_epochs']
    step_count = 0
    
    angle_weight = train_config['angle_loss_weight']
    
    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        frcnn_angles_losses = []
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
                
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']
            loss += angle_weight * batch_losses['loss_angles']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
            frcnn_angles_losses.append(batch_losses['loss_angles'].item())

            loss.backward()
            optimizer.step()
            step_count +=1
            
        print('Finished epoch {}'.format(i))
        # torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                # 'classification_oriented_tv_frcnn_r50fpn_' + train_config['ckpt_name']))
        # save_checkpoint(faster_rcnn_model, optimizer, i, f'./checkpoints/classification_oriented_checkpoint_epoch_{i}.pth')
        
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        loss_output += ' | FRCNN Angles Loss : {:.4f}'.format(np.mean(frcnn_angles_losses))
        print(loss_output)
        
        
    print('Done Training...')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/st.yaml', type=str)
    args = parser.parse_args()
    train(args)
    
    
# '''
# Layer 0 output shape: torch.Size([1, 256, w, h])
# Layer 1 output shape: torch.Size([1, 256, w/2, h/2])
# Layer 2 output shape: torch.Size([1, 256, w/4, h/4])
# Layer 3 output shape: torch.Size([1, 256, w/8, h/8])
# pool output shape: torch.Size([1, 256, w/16, h/16])
# '''