from typing import List, Optional
import os
import sys
import yaml
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def yaml_parser(
    path: Optional[str] = None,
    data: Optional[str] = None,
    loader: yaml.SafeLoader = yaml.SafeLoader
) -> dict:
    if path:
        with open(r"{}".format(path)) as file:
            return yaml.load(file, Loader=loader)

    elif data:
        return yaml.load(data, Loader=loader)

    else:
        raise ValueError('Either a path or data should be defined as input')


def load_data_openimages(
    loader_list: str,
    downloader_path: str,
    download_folder: str,
    num_processes: int,
) -> None:
    python_path = sys.executable
    try:
        import boto3
        import botocore
    except:
        print("installing boto3 and botocore for downloader by openimages ...")
        os.system("{} -m pip install -q boto3".format(python_path))
        os.system("{} -m pip install -q botocore".format(python_path))

    os.system("{} {} {} --download_folder={} --num_processes={}".format(
        python_path, downloader_path, loader_list,
        download_folder, num_processes
    ))


def get_dataset(
    root: str,
    part: str,
    class_names: Optional[List[str]] = None,
    num_processes: int = 5,
    remove_empty_imgs: bool = True
) -> List[str]:
    if not os.path.exists(root):
        os.makedirs(root)

    source_dict = yaml_parser('./conf/conf_dataset.yaml')
    check_list = [
        'cls_desc',
        'test_bbox',
        'validation_bbox',
        'train_bbox',
        'test_ann',
        'validation_ann',
        'train_ann']

    for key, value in source_dict.items():
        if key in check_list:
            path_i = value.split('/')[-1]
            if not os.path.exists(os.path.join(root, path_i)):
                os.system("wget {} -O {}/{} -q".format(
                    value, root, path_i))

    annos = source_dict[part + "_ann"].split('/')[-1]
    img_id = pd.read_csv(os.path.join(root, annos))
    img_id.loc[:, "Subset"] = part
    img_id = img_id[~img_id["LabelName"].isnull()]
    class_names_df = pd.read_csv(
        os.path.join(root, source_dict['cls_desc'].split('/')[-1]),
        names=["LabelName", "ClassName"])

    img_id = pd.merge(img_id, class_names_df, on="LabelName", how="left")
    img_id.drop(["LabelName", "Source"], axis=1, inplace=True)

    saving_path = os.path.join(root, part + "_images_list.txt")
    downloader_path = os.path.join(".", root, "downloader.py")
    download_folder = os.path.join(".", root, part)
    if class_names:
        img_id = img_id[img_id["ClassName"].isin(class_names)]

    bboxs = source_dict[part + "_bbox"].split('/')[-1]
    img_bboxs_df = pd.read_csv(
        os.path.join(root, bboxs),
        usecols=["ImageID", "XMin", "XMax", "YMin", "YMax"])
    img_id = pd.merge(img_id, img_bboxs_df, on="ImageID", how="left")
    if remove_empty_imgs:
        img_id = img_id[~((img_id["XMin"].isnull()) | (img_id["XMax"].isnull())
            | (img_id["YMin"].isnull()) | (img_id["YMax"].isnull()))]

    img_id["bbox"] = img_id[img_id.columns[4:]].apply(
        lambda x: [i for i in x],
        axis=1)
    img_id.drop(["XMin", "XMax", "YMin", "YMax"], inplace=True, axis=1)
    img_id = img_id[["ImageID", "bbox"]].groupby("ImageID")["bbox"].apply(list).reset_index()

    if not os.path.exists(os.path.join(root, part)):
        downloader_ulr = source_dict['downloader_ulr']
        os.system("wget {} -P {}".format(downloader_ulr, root))
        image_id_file = img_id["ImageID"]
        image_id_file = image_id_file.apply(lambda x: part + '/' + str(x))
        image_id_file.to_csv(saving_path, header=False, index=False)
        load_data_openimages(
            loader_list=saving_path,
            downloader_path=downloader_path,
            download_folder=download_folder,
            num_processes=num_processes)
        os.remove(downloader_path)

    return img_id.set_index("ImageID")["bbox"].to_dict()


def collate_fn(batch):
    """collate_fn needs for batch"""
    return tuple(zip(*batch))


class OpenImagesDataset(Dataset):
    def __init__(
        self,
        root: str,
        part: str,
        class_names: Optional[List[str]] = None,
        transforms: Optional[transforms.Compose] = None
    ) -> None:
        self.root = root
        self.part = part
        self.data_dict = get_dataset(
            root=root,
            part=part,
            class_names=class_names)

        self.imgs = list(self.data_dict.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images and masks
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, self.part, img_name + ".jpg")
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        num_objs = len(self.data_dict[img_name])
        boxes = []
        for i in range(num_objs):
            bbox = self.data_dict[img_name][i]
            xmin = bbox[0] * w
            xmax = bbox[1] * w
            ymin = bbox[2] * h
            ymax = bbox[3] * h
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # is crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_data_dict(self):
        return self.data_dict
