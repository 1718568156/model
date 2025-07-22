import os
import shutil # 导入用于文件移动的库
from PIL import Image

def pre_clean_dataset_safe(root_folder, quarantine_folder_name='_quarantine'):
    """
    对整个数据集目录执行安全的、非破坏性的预清洗。
    不符合要求的图片会被移动到隔离文件夹中。
    """
    # 在数据集根目录下创建总的隔离文件夹
    quarantine_base_path = os.path.join(root_folder, quarantine_folder_name)
    if not os.path.exists(quarantine_base_path):
        os.makedirs(quarantine_base_path)
    print(f"已创建或确认隔离文件夹: {quarantine_base_path}")

    # 遍历所有类别子文件夹
    for class_name in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_name)
        # 确保处理的是文件夹，并且不是我们刚创建的隔离文件夹
        if os.path.isdir(class_path) and class_name != quarantine_folder_name:
            
            # 在隔离区为当前类别创建对应的子文件夹
            quarantine_class_path = os.path.join(quarantine_base_path, class_name)
            if not os.path.exists(quarantine_class_path):
                os.makedirs(quarantine_class_path)

            print(f"\n{'='*50}\n正在处理类别文件夹: {class_path}\n{'='*50}")
            
            # --- 1. 按文件大小过滤 (安全模式) ---
            filter_images_by_size_safe(class_path, quarantine_class_path)
            
            # --- 2. 按像素尺寸/宽高比过滤 (安全模式) ---
            filter_images_by_dimensions_safe(class_path, quarantine_class_path)

def filter_images_by_size_safe(directory_path, quarantine_path, min_size_kb=10, max_size_kb=5000):
    print(f"\n--- 正在按文件大小过滤 (安全模式)... ---")
    moved_count = 0
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            try:
                file_size_kb = os.path.getsize(filepath) / 1024
                if file_size_kb < min_size_kb or file_size_kb > max_size_kb:
                    # 移动文件而不是删除
                    shutil.move(filepath, os.path.join(quarantine_path, filename))
                    print(f"  > 移动[尺寸异常:{file_size_kb:.2f}KB]: {filename}")
                    moved_count += 1
            except Exception as e:
                print(f"处理 {filename} 出错: {e}")
    print(f"文件大小过滤完成, 移动 {moved_count} 个文件。")


def filter_images_by_dimensions_safe(directory_path, quarantine_path, min_width=100, min_height=100, aspect_ratio_range=(0.5, 2.0)):
    print(f"\n--- 正在按像素尺寸/宽高比过滤 (安全模式)... ---")
    moved_count = 0
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    reason = None
                    if width < min_width or height < min_height:
                        reason = "尺寸过小"
                    elif not (aspect_ratio_range[0] <= width / height <= aspect_ratio_range[1]):
                        reason = "宽高比异常"
                    if reason:
                        img.close()
                        # 移动文件而不是删除
                        shutil.move(filepath, os.path.join(quarantine_path, filename))
                        print(f"  > 移动[{reason}:{width}x{height}]: {filename}")
                        moved_count += 1
            except Exception:
                try:
                    # 移动无法打开的损坏文件
                    shutil.move(filepath, os.path.join(quarantine_path, filename))
                    print(f"  > 移动[无法打开]: {filename}")
                    moved_count += 1
                except Exception as e:
                    print(f"移动损坏文件 {filename} 时出错: {e}")
    print(f"像素尺寸过滤完成, 移动 {moved_count} 个文件。")


# --- 使用方法 ---
if __name__ == '__main__':
    # 设置你的数据集根目录
    dataset_root = 'garbage_dataset'
    
    if os.path.exists(dataset_root):
        pre_clean_dataset_safe(dataset_root)
        print(f"\n{'='*50}\n所有文件夹预清洗完成！不合格文件已移至 _quarantine 文件夹。\n{'='*50}")
    else:
        print(f"错误：数据集根目录不存在 -> {dataset_root}")