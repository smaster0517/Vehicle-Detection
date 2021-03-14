import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import seaborn as sns


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def visualize_from_dataset(img):
    img_value, img_boxes = img[0], img[1]["boxes"]
    img_value = img_value.permute(1, 2, 0).numpy()
    img = img_value.copy()
    for i in range(len(img_boxes)):
        x_min, y_min, x_max, y_max = img_boxes[i]
        image = cv2.rectangle(img, 
                              (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 2)
    # image = image.get()
    show_image(image)


def show_image(image, figsize=(16, 9), reverse=True):
    plt.figure(figsize=figsize)
    if reverse:
        plt.imshow(image[..., ::-1])
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()


def val_transform(img):
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img)
    return img_tensor.unsqueeze(0)


def visualize_prediction_plate(
    idx,
    dataset,
    model,
    device='cuda',
    verbose=True,
    thresh=0.0,
    n_colors=None,
    id_to_name=None
) -> any:
    filename_to_bbox_dict = dataset.get_data_dict()
    file_name = list(filename_to_bbox_dict.keys())[idx]
    file_path = os.path.join('.', 'data', dataset.part, file_name + '.jpg')

    img = Image.open(file_path)
    img_tensor = val_transform(img)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor.to(device))
    prediction = predictions[0]

    if n_colors is None:
        n_colors = model.roi_heads.box_predictor.cls_score.out_features

    palette = sns.color_palette(None, n_colors)

    img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    image = img
    for i in range(len(prediction['boxes'])):
        x_min, y_min, x_max, y_max = map(int, prediction['boxes'][i].tolist())
        label = int(prediction['labels'][i].cpu())
        score = float(prediction['scores'][i].cpu())
        name = id_to_name[label]
        color = palette[label]
        if verbose:
            if score > thresh:
                print('Class: {}, Confidence: {}'.format(name, score))
        if score > thresh:
            image = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), np.array(color) * 255, 2)
            cv2.putText(image, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(color) * 255, 2)
    show_image(image)
    return prediction
