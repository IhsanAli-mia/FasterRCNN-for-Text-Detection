import yaml
import torch
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader
from dataset.st import SceneTextDataset
from detection.faster_rcnn import FastRCNNPredictor
import detection
from detection.transform import GeneralizedRCNNTransform
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Union


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
val = SceneTextDataset('', root_dir=dataset_config['root_dir'])
val_dataset = DataLoader(val,batch_size=1,shuffle=True,num_workers=6,collate_fn=collate_function)


faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                    min_size=600,
                                                    max_size=1000,
)
faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
    faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
    num_classes=dataset_config['num_classes'])

val_iter = iter(val_dataset)
val_batch = next(val_iter)
val_batch = next(val_iter)
val_batch = next(val_iter)
val_batch = next(val_iter)
val_batch = next(val_iter)
val_batch = next(val_iter)
val_images,targets,_ = val_batch
val_images = [im.float() for im in val_images]

heatmaps_per_epoch = {}

min_size=600
max_size=1000
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

original_image_sizes: List[Tuple[int, int]] = []
for img in val_images:
    val = img.shape[-2:]
    torch._assert(
        len(val) == 2,
        f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
    )
    original_image_sizes.append((val[0], val[1]))

# print(val_images[0].size())

dim = val_images[0].size()
width,height = dim[1],dim[2]


for target in targets:
            target['boxes'] = target['bboxes'].float().to(device)
            del target['bboxes']
            target['labels'] = target['labels'].long().to(device)
# targets = targets.to(device)
val_images,_ = transform(val_images,targets)
val_images = val_images.to(device)
# # print(val_images)

# HEATMAP

# video_frames_heatmap = {0:[]}


# for i in range(15):
    
#     checkpoint_path = f'./checkpoints/checkpoint_epoch_{i}.pth'
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     faster_rcnn_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
#     faster_rcnn_model.to(device)
#     faster_rcnn_model.train()

#     with torch.no_grad():
#             features = faster_rcnn_model.backbone(val_images.tensors)
#             features_list = list(features.values())
#             objectness_maps, _ = faster_rcnn_model.rpn.head(features_list)
#             # proposals, _ = faster_rcnn_model.rpn(val_images, features, targets)
#     heatmaps_per_epoch[i] = [obj.sigmoid().mean(dim=1).cpu().numpy() for obj in objectness_maps]
        
#     for img_idx in range(1):  # Loop over images
#         fig, axes = plt.subplots(1, len(heatmaps_per_epoch[i]), figsize=(15, 5))
#         for level, heatmap in enumerate(heatmaps_per_epoch[i]):  # Loop over FPN levels
#             axes[level].imshow(heatmap[img_idx], cmap='jet')
#             axes[level].set_title(f"Epoch {i} | FPN Level {level}")
#             axes[level].axis("off")
#         plt.tight_layout()
#         filename = f"./heatmap_images/heatmap_img{img_idx}_epoch{i}.png"
#         plt.savefig(filename)
#         plt.close()
#         frame = cv2.imread(filename)
#         # frame = frame.astype(np.uint8)
#         # cv2.cvtColor(frame,cv2.COLOR_)
#         video_frames_heatmap[img_idx].append(frame)
        
# fps = 5  # frames per second

# # print(video_frames_heatmap)

# height, width = video_frames_heatmap[0][0].shape[:2]
# output_filename = 'objectness_heatmaps_animation_2.mp4'

# # Create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
# out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# # Write each image to the video
# for img in video_frames_heatmap[0]:
#     out.write(img)

# # Release the VideoWriter
# out.release()

# print(f"Animation saved as {output_filename}")

# video_frames_proposals = {0:[]}
# proposals_over_epochs = []

# def visualize_proposals(epoch):
#     plt.clf()
#     fig, axes = plt.subplots(1, len(val_images), figsize=(15, 5))

#     for img_idx, ax in enumerate(axes):
#         image = val_images[img_idx].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
#         ax.imshow(image, cmap='gray')

#         # Draw bounding boxes
#         for box in proposals_over_epochs[epoch][img_idx][:10]:  # Show top 10 proposals
#             x1, y1, x2, y2 = box
#             rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)

#         ax.set_title(f"Epoch {epoch}")
#         ax.axis("off")
        
# val_images_cpu = val_images.tensors.cpu().numpy()

# for i in range(15):
    
#     checkpoint_path = f'./checkpoints/checkpoint_epoch_{i}.pth'
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     faster_rcnn_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
#     faster_rcnn_model.to(device)
#     faster_rcnn_model.train()

#     with torch.no_grad():
#         features = faster_rcnn_model.backbone(val_images.tensors)
#         # features_list = list(features.values())
#         # objectness_maps, _ = faster_rcnn_model.rpn.head(features_list)
#         proposals, _ = faster_rcnn_model.rpn(val_images, features, targets)
#     proposals_over_epochs.append([proposal.cpu().numpy() for proposal in proposals])
    
#     image = val_images_cpu[0].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
#     proposals_np = proposals[0].cpu().numpy()  # Convert to NumPy

#     # Plot the image with bounding boxes
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(image, cmap='gray')

#     for box in proposals_np[:10]:  # Show top 10 proposals
#         x1, y1, x2, y2 = box
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#     ax.set_title(f"Epoch {i}")
#     ax.axis("off")

#     fig.canvas.draw()
#     img_np = np.array(fig.canvas.renderer.buffer_rgba())  # (H, W, 4) RGBA format
#     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)  # Convert to BGR

#     video_frames_proposals[0]=img_np  # Store in list for animation

#     # Save frame for reference
#     cv2.imwrite(f"./proposal_frames/proposal_frame_epoch_{i}.png", img_np)

#     plt.close()
    
# fps = 5  # frames per second

# # print(video_frames_proposals)

# height, width = video_frames_proposals[0][0].shape[:2]
# # print(height, width)
# output_filename = 'proposal_animation_2.mp4'

# # Create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
# out = cv2.VideoWriter(output_filename, fourcc, fps, (800,800))

# # Write each image to the video

# for i in range(15):
#     filename = f"./proposal_frames/proposal_frame_epoch_{i}.png"
#     frame = cv2.imread(filename)
#     print(frame.shape)
#     out.write(frame)

# # Release the VideoWriter
# out.release()

# print(f"Animation saved as {output_filename}")

for i in range(14,15):
    
    checkpoint_path = f'./checkpoints/checkpoint_epoch_{i}.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    faster_rcnn_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    faster_rcnn_model.to(device)
    faster_rcnn_model.train()

    with torch.no_grad():
        features = faster_rcnn_model.backbone(val_images.tensors)
        features_list = list(features.values())  # Convert feature dict to list
        anchors = faster_rcnn_model.rpn.anchor_generator(val_images, features_list)  # Generate anchors

        labels, matched_gt_boxes = faster_rcnn_model.rpn.assign_targets_to_anchors(anchors, targets)

        # # Convert images to NumPy format for visualization
        image = val_images.tensors[0].cpu().numpy().transpose(1, 2, 0)  # Convert (C, H, W) → (H, W, C)

        # # Convert anchors and labels to NumPy
        anchors_np = anchors[0].cpu().numpy()  # Shape: (num_anchors, 4)
        labels_np = labels[0].cpu().numpy()  # Shape: (num_anchors,)

        # # # Select up to 10 positive (label == 1) and 10 negative (label == 0) anchors
        pos_indices = np.where(labels_np == 1)[0]  # First 10 positive anchors
        neg_indices = np.where(labels_np == 0)[0]  # First 10 negative anchors

        if len(pos_indices) > 10:
            pos_indices = np.random.choice(pos_indices, 10, replace=False)
        if len(neg_indices) > 10:
            neg_indices = np.random.choice(neg_indices, 10, replace=False)
        # print(pos_indices)
        # print(neg_indices)

        # print("Everything done before plotting")
        
        # Plot the image
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='gray')

        # Plot positive anchors (Green)
        for idx in pos_indices:
            x1, y1, x2, y2 = anchors_np[idx]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Plot negative anchors (Red)
        for idx in neg_indices:
            x1, y1, x2, y2 = anchors_np[idx]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.set_title("Positive Anchors (Green) | Negative Anchors (Red) | All Anchors (Blue)")
        ax.axis("off")
        
        # # print("Everything done before drawing")

        # # Convert Matplotlib figure to OpenCV image
        fig.canvas.draw()
        img_np = np.array(fig.canvas.renderer.buffer_rgba())  # RGBA format
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)  # Convert to BGR for OpenCV

        # print("One epoch over")

        # Save the visualization
        cv2.imwrite(f"./bounding_box_visualised/bounding_box_assignments_{i}_3.png", img_np)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)

        # Visualizing anchors for each feature map
        # print(anchors_np)
        
        num_anchors = min(len(anchors_np), 500)  # Limit to 500 for clarity
        sampled_anchors = anchors_np[np.random.choice(len(anchors_np), num_anchors, replace=False)]

        for box in sampled_anchors:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=0.5, edgecolor='blue', facecolor='none', alpha=0.3
                )
                ax.add_patch(rect)

        ax.set_title("Subset of Anchors Across Feature Maps")
        ax.axis("off")

        # Save the figure
        save_path = "./bounding_box_visualised/anchors_image_3.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


# for i in range(15):
    
#     checkpoint_path = f'./checkpoints/checkpoint_epoch_{i}.pth'
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     faster_rcnn_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
#     faster_rcnn_model.to(device)
#     faster_rcnn_model.train()

#     with torch.no_grad():
#         features = faster_rcnn_model.backbone(val_images.tensors)
#         features_list = list(features.values())
#         all_proposals,_ = faster_rcnn_model.rpn(val_images,features,targets)
        
#         num_proposals = min(len(all_proposals[0]), 50)  
#         selected_indices = torch.randint(0, len(all_proposals[0]), (num_proposals,))
#         subset_proposals = [all_proposals[0][selected_indices]] 
        
#         faster_rcnn_model.eval()
#         detections,_= faster_rcnn_model.roi_heads(features,subset_proposals,val_images.image_sizes,targets)
#         detections = faster_rcnn_model.transform.postprocess(detections,val_images.image_sizes,original_image_sizes)
    
#     image_np = val_images.tensors[0].cpu().numpy().transpose(1, 2, 0)
    
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(image_np)

#     # Plot RPN Proposals (Blue)
    
#     for box in subset_proposals[0].cpu().numpy():
#         x1, y1, x2, y2 = box
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                  linewidth=1, edgecolor='blue', facecolor='none', alpha=0.5)
#         ax.add_patch(rect)

#     # # Plot Final Detections (Red) with Class Scores
    
#     for box, score in zip(detections[0]['boxes'], detections[0]['scores']):
#         # print(box,score)
#         x1, y1, x2, y2 = box.cpu().numpy()
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                  linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
#         ax.add_patch(rect)

#         # Display classification score
#         ax.text(x1, y1 - 5, f"{score:.2f}", color='white', fontsize=8, 
#                 bbox=dict(facecolor='red', alpha=0.5))

#     ax.set_title("RPN Proposals (Blue) vs ROIHead Predictions (Red)")
#     ax.axis("off")

#     save_path = f"./roi_proposals/bounding_box_{i}_.png"
    
#     # Save the figure
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()

#     print(f"Saved visualization to: {save_path}")
    
# # print(val_images.size())
        
# fps = 5

# # height, width = 1729, 1919
# # # print(height, width)
# output_filename = 'bounding_box_proposals_1.mp4'

# print(width,height)

# # Create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
# out = cv2.VideoWriter(output_filename, fourcc, fps, (width,height))


# for i in range(1,15):
#     filename = f"./roi_proposals/bounding_box_{i}_.png"
#     frame = cv2.imread(filename)
#     print(frame.shape)
#     out.write(frame)

# # Release the VideoWriter
# out.release()