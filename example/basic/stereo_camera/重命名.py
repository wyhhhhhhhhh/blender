import os

def rename_images(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    files.sort()  # 按文件名排序

    # 重新命名文件
    for idx, file_name in enumerate(files, start=1):
        old_path = os.path.join(directory, file_name)
        new_name = f"{idx:04d}.npy"  # 按顺序重命名为 0001.png, 0002.png, ...
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

if __name__ == "__main__":
    image_dir = "left"  # 替换为你的文件夹路径
    image_dir2 = "right"  # 替换为你的文件夹路径
    rename_images(image_dir)
    rename_images(image_dir2)
