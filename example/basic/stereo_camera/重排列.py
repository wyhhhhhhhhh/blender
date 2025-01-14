import os

def rename_images(directory, start_old, end_old, start_new):
    old_numbers = range(start_old, end_old + 1)
    new_numbers = range(start_new, start_new + len(old_numbers))

    for old_num, new_num in zip(old_numbers, new_numbers):
        old_name = f"{old_num:04d}.png"
        new_name = f"{new_num:04d}.png"

        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"File {old_name} does not exist, skipping.")

if __name__ == "__main__":
    image_dir = "left"
    image_dir2 = "right"
    rename_images(image_dir, start_old=1, end_old=33, start_new=445)
    rename_images(image_dir2, start_old=1, end_old=33, start_new=445)
