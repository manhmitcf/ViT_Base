import os
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
import pandas as pd

csv_path = r"data\train.csv"
df = pd.read_csv(csv_path)
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor()])
new_data = []
for index, row in df.iterrows():
    root_dir_image = "data/images"
    Path = row["Path"]
    Fish_Name = row["Fish Name"]
    Label = row["Label"]
    image_path = os.path.join("data", "images", Path)
    image = read_image(root_dir_image + "/" + Path)
    image = transforms.ToPILImage()(image)
    augmented_img = augment_transform(image)
    new_img_name = f"{os.path.basename(Path).split('.')[0]}_aug.jpg"
    save_path = f"data/images_aug/{new_img_name}"
    save_image(augmented_img, save_path)
    new_data.append([Fish_Name, Label, Path])
df_new = pd.DataFrame(new_data, columns=["Fish Name", "Label", "Path"])
df_final = pd.concat([df, df_new], ignore_index=True)
df_final.to_csv("data/train_aug.csv", index=False)
print("Tạo ảnh augmentation & cập nhật CSV thành công! 🎉")