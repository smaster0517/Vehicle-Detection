import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torchvision import transforms
import cv2


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


def visualize_prediction(file, model, device='cuda', verbose=True, thresh=0.0, n_colors=None):
    path_to_images = '/content/test_images/test_images'
    file_path = os.path.join(path_to_images, file, 'images', file + '.png')
    img = Image.open(file_path)
    transform_to_tensor = transforms.ToTensor()
    img_tensor = transform_to_tensor(img).unsqueeze(0)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor.to(device))[0]
    visualize(image=img, mask=predictions.cpu().squeeze(0))


def visualize_from_dataset(img):
    img_value, img_boxes = img[0], img[1]["boxes"]
    img_value = img_value.permute(1, 2, 0).numpy() * 255
    for i in range(len(img_boxes)):
        x_min, y_min, x_max, y_max = img_boxes[i]
        image = cv2.rectangle(
            img_value.astype(dtype=np.uint8),
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2)
    image = image.get()
    show_image(image)


def show_image(image, figsize=(16, 9), reverse=True):
    plt.figure(figsize=figsize)
    if reverse:
        plt.imshow(image[..., ::-1])
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()

