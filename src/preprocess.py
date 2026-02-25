import os, pickle
from collections import OrderedDict
import pandas as pd

# 1. Copy/move all ISIC 2024 images into data/isic2024/final_image/
# 2. Build image dict
img_dir = "/home/mecha109/PanDerm/Evaluation_datasets/processed_images_224x224"
image_dict = OrderedDict()
for f in sorted(os.listdir(img_dir)):
    key = os.path.splitext(f)[0]  # isic_id without extension
    image_dict[key] = f

with open("/home/mecha109/Github/MONET/data/isic2024/image_dict.pkl", "wb") as f:
    pickle.dump({"images": image_dict}, f)

# 3. Prepare metadata with a label column
meta = pd.read_csv("/home/mecha109/PanDerm_backup/Evaluation_datasets/ISIC2024/train-metadata.csv")
meta = meta.set_index("isic_id")
# The ISIC 2024 challenge uses "target" column (0=benign, 1=malignant)
meta.to_csv("/home/mecha109/Github/MONET/data/isic2024/final_metadata_all.csv")