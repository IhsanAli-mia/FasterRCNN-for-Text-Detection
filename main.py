import yaml
import torch
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader
from dataset.st import SceneTextDataset
from detection.faster_rcnn import FastRCNNPredictor
import detection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config_path = 'config/st.yaml'

def collate_function(data):
    return tuple(zip(*data))

with open(config_path, 'r') as file:
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

if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
st = SceneTextDataset('train', root_dir=dataset_config['root_dir'])

train_dataset = DataLoader(st,
                           batch_size=1,
                           shuffle=True,
                           num_workers=8,
                           collate_fn=collate_function)

# val_dataset = DataLoader(val,batch_size=1,shuffle=True,num_workers=6,collate_fn=collate_function)
faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                    min_size=600,
                                                    max_size=1000,
)
faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
    faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
    num_classes=dataset_config['num_classes'])

checkpoint_path = './checkpoints/oriented_checkpoint_epoch_15.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

faster_rcnn_model.load_state_dict(checkpoint['model_state_dict'])
faster_rcnn_model.to(device)
faster_rcnn_model.train()

rpn_classification_losses = []
rpn_localization_losses = []
frcnn_classification_losses = []
frcnn_localization_losses = []
frcnn_angles_losses = []


images, targets, _ = next(iter(train_dataset))

for target in targets:
    target['boxes'] = target['bboxes'].float().to(device)
    del target['bboxes']
    target['labels'] = target['labels'].long().to(device)
image = [im.float().to(device) for im in images]

batch_losses = faster_rcnn_model(image, targets)
# batch_losses = batch_losses[0] if isinstance(batch_losses, list) else batch_losses

# print(batch_losses.keys())


rpn_classification_losses.append(batch_losses['loss_objectness'].item())
rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
frcnn_angles_losses.append(batch_losses['loss_angles'].item())

loss_output = ''
loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
loss_output += ' | FRCNN Angles Loss : {:.4f}'.format(np.mean(frcnn_angles_losses))
print(loss_output)