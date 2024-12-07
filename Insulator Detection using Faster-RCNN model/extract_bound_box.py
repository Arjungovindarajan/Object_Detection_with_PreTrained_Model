import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import torchvision.transforms as T
# Function to parse XML files and extract bounding boxes
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes

# Custom dataset class to handle images and their respective annotations
class InsulatorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or T.ToTensor()
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "Images"))))
        self.xmls = list(sorted(os.listdir(os.path.join(root_dir, "Annotations"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Apply transform (convert image to tensor)
        img = self.transform(img)

        xml_path = os.path.join(self.root_dir, "annotations", self.xmls[idx])
        boxes = parse_annotation(xml_path)

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)  # 1 for insulator
        
        return img, target
