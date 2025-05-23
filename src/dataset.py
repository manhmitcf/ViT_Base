import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FishDatasetWithAugmentation(Dataset):
    def __init__(self, csv_file, img_dir, img_dir_aug, transform=None, aug_transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.aug_transform = aug_transform
        self.img_dir_aug = img_dir_aug
        self.labels = {"Highly Fresh" : 0, "Fresh" : 1, "Not Fresh": 2}  
        # Kiểm tra dữ liệu đầu vào
        if not os.path.exists(img_dir) or not os.path.exists(img_dir_aug):
            raise FileNotFoundError(f"Thư mục ảnh '{img_dir}' hoặc '{img_dir_aug}' không tồn tại.")
        if self.data.empty:
            raise ValueError(f"File CSV '{csv_file}' không chứa dữ liệu.")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        check = False
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 2])
        if not os.path.exists(img_name):
            img_name = os.path.join(self.img_dir_aug, self.data.iloc[idx, 2])
            if not os.path.exists(img_name):
                check = True
        if check:
            img_name = os.path.join(self.img_dir, self.data.iloc[idx, 2])
            img_name = img_name.replace('_5', '_#')
            if not os.path.exists(img_name):
                img_name = os.path.join(self.img_dir_aug, self.data.iloc[idx, 2])
                img_name = img_name.replace('_5', '_#')
                if not os.path.exists(img_name):
                    raise FileNotFoundError(f"Không tìm thấy ảnh '{img_name}'.")
        try:
            image = Image.open(img_name).convert('RGB')  # Đọc và chuyển đổi ảnh sang RGB
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy ảnh '{img_name}'.")


        label = self.data.iloc[idx, 1]
        if label not in self.labels:
            raise ValueError(f"Nhãn '{label}' không hợp lệ. Phải là một trong {list(self.labels.keys())}.")
        label = self.labels[label]
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        elif self.aug_transform:
            image = self.aug_transform(image)
        else:
            raise ValueError("Cả transform và aug_transform đều là None. Ít nhất một trong hai phải được cung cấp.")

        return image, label
    
basic_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

aug_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=3),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03), shear=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


