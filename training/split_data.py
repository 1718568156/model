import os
import shutil
import random

def split_dataset(source_folder, dest_folder, train_ratio=0.8):
    """
    将ImageFolder格式的数据集划分为训练集和验证集。
    """
    # --- 1. 创建目标文件夹结构 ---
    if os.path.exists(dest_folder):
        print(f"警告：目标文件夹 {dest_folder} 已存在，将清空并重新创建。")
        shutil.rmtree(dest_folder)
        
    train_path = os.path.join(dest_folder, 'train')
    val_path = os.path.join(dest_folder, 'val')
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    
    print(f"已创建目标文件夹: {train_path} 和 {val_path}")

    # --- 2. 遍历每个类别并划分 ---
    for class_name in os.listdir(source_folder):
        source_class_path = os.path.join(source_folder, class_name)
        
        if os.path.isdir(source_class_path):
            print(f"\n正在处理类别: {class_name}")
            
            train_class_path = os.path.join(train_path, class_name)
            val_class_path = os.path.join(val_path, class_name)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(val_class_path, exist_ok=True)
            
            images = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)
            
            split_point = int(len(images) * train_ratio)
            train_images = images[:split_point]
            val_images = images[split_point:]
            
            # --- 3. 复制文件到新目录 ---
            print(f"  > 复制 {len(train_images)} 张图片到训练集...")
            for img in train_images:
                shutil.copy(os.path.join(source_class_path, img), os.path.join(train_class_path, img))
                
            print(f"  > 复制 {len(val_images)} 张图片到验证集...")
            for img in val_images:
                shutil.copy(os.path.join(source_class_path, img), os.path.join(val_class_path, img))

    print("\n数据集划分完成！")


# --- 使用方法 ---
if __name__ == '__main__':
    # 脚本与 'garbage_dataset' 在同一目录下，直接使用相对路径
    source_dataset_dir = 'garbage_dataset'
    
    # 在当前目录下创建划分后的数据集文件夹
    split_dataset_dir = 'split_garbage_dataset'
    
    if os.path.exists(source_dataset_dir):
        split_dataset(source_dataset_dir, split_dataset_dir, train_ratio=0.8)
    else:
        print(f"错误：原始数据集文件夹 '{source_dataset_dir}' 不在当前目录中。")