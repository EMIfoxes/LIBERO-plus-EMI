from pathlib import Path
import json
import os
root = Path('/home/lxx/project/DreamVLA/datasets/libero_10_converted')
episodes_path = root/'episodes'
data_info_path = root/'data_info'
json_name = 'libero_10_converted.json'
# 创建路径
dir_path = str(data_info_path)          # 转成字符串，纯 os 风格
os.makedirs(dir_path, exist_ok=True) 

result = []
for ep_dir in sorted(episodes_path.glob('[0-9]'*6)):   # 匹配 6 位数字文件夹
    step_path = ep_dir/'steps'
    folder_num = sum(1 for p in step_path.iterdir() if p.is_dir())
    result.append([ep_dir.name, folder_num])

out_file = data_info_path / json_name
out_file.write_text(json.dumps(result))
print('已生成', out_file.resolve())