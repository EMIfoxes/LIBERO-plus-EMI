import tensorflow_datasets as tfds
import cv2
import numpy as np
from collections import defaultdict
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

name = 'libero_object'
data_dir = '/media/lxx/新加卷/DL_datasets/rlds/libero_plus_data_4suite'
builder = tfds.builder(name, data_dir=data_dir)

split_info = builder.info.splits['train']
print('轨迹数 =', split_info.num_examples) # goal 有4243个episodes | object 有4520个episodes


ds_episode = builder.as_dataset(split='train', shuffle_files=False) # split='train' split='train[:100]'

task_cat_cnt = defaultdict(lambda: defaultdict(int))
for episode in ds_episode:                # 第 1 层：遍历每条轨迹
    file_path_bytes = episode['episode_metadata']['file_path'].numpy()  # 返回 bytes
    file_path_str = file_path_bytes.decode('utf-8')                     # 转 str
    file_path = Path(file_path_str) 
    parts = file_path.parts
    try:
        category = parts[parts.index('pro_data') + 1]   # language / vision / …
        task = file_path.stem                       # pick_up_the_ketchup_and_place_it_in_the_basket_demo
    except ValueError:
        continue   # 路径格式不对就跳过

    task_cat_cnt[task][category] += 1

   

# 输出结果
print(f'共 {len(task_cat_cnt)} 个不同任务')
for task, cats in task_cat_cnt.items():
    print(f'{task}: {len(cats)} 个类别 → {sorted(cats)}')
    for key,value in cats.items():
        print(f'  {key}: {value} 个轨迹')

tasks = list(task_cat_cnt.keys())
fig, axes = plt.subplots((len(tasks) + 5) // 4, 4, figsize=(18, 3 * ((len(tasks) + 5) // 5)))
axes = axes.flatten() if len(tasks) > 1 else [axes]

for ax, task in zip(axes, tasks):
    cats = list(task_cat_cnt[task].keys())
    counts = list(task_cat_cnt[task].values())

    ax.bar(cats, counts, color='skyblue')
    ax.set_xticks(cats)
    ax.set_xticklabels(cats, fontsize=8, ha='center') # rotation=45,
    ax.set_title(task[:80] + '...' if len(task) > 30 else task, fontsize=8)
    # ax.set_ylabel('nums')
    # ax.set_xlabel('category')
    # 在柱子上方标数字
    for c, v in zip(cats, counts):
        ax.text(c, v + 0.2, str(v), ha='center', va='bottom')

# 隐藏多余子图
for j in range(len(tasks), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('/media/lxx/Elements/project/VLA-Adapter/demo.png')