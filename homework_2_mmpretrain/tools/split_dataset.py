import os
import random
import shutil

# 设置目录路径和划分比例
directory = "../data/fruit30_train"
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# 创建目标文件夹
train_dir = "../data/train"
val_dir = "../data/validation"
test_dir = "../data/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历每个类别目录
for class_name in os.listdir(directory):
    class_dir = os.path.join(directory, class_name)
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        random.shuffle(images)

        # 计算划分数量
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)
        num_test = num_images - num_train - num_val

        # 划分图片并复制到对应的目标文件夹
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        for image in train_images:
            src_path = os.path.join(class_dir, image)
            dst_path = os.path.join(train_dir, class_name, image)
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            shutil.copy(src_path, dst_path)

        for image in val_images:
            src_path = os.path.join(class_dir, image)
            dst_path = os.path.join(val_dir, class_name, image)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            shutil.copy(src_path, dst_path)

        for image in test_images:
            src_path = os.path.join(class_dir, image)
            dst_path = os.path.join(test_dir, class_name, image)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
            shutil.copy(src_path, dst_path)
