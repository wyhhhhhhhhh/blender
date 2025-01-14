import shutil

# 原始文件名
original_filename = '0001.json'
# 复制的文件数量
num_copies = 499

# 循环复制文件
for i in range(2, num_copies + 2):  # 从0002开始，到0501结束
    new_filename = f'{i:04}.png'  # 格式化文件名为4位数，不足的前面补0
    shutil.copyfile(original_filename, new_filename)
    print(f'Copied {original_filename} to {new_filename}')  # 打印复制信息