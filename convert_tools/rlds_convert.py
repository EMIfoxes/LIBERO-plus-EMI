import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import os
import h5py
from PIL import Image
from tqdm import tqdm
from itertools import islice
import argparse

def main(args):
    os.makedirs(Path(args.outpu_dir)/'episodes', exist_ok=True) # 创建文件夹
    builder = tfds.builder(args.suit_name, data_dir=args.data_dir)
    split_info = builder.info.splits['train']
    print('episodes =', split_info.num_examples)                        # goal 有4243个episodes | object 有4520个episodes
    ds_episode = builder.as_dataset(split='train', shuffle_files=False) # 返回一个 tf.data.Dataset 对象
    episode_index = 0

    if args.num_examples is not None:
        num_examples = args.num_examples
    else:
        num_examples = split_info.num_examples

    for episode in tqdm(islice(ds_episode, num_examples), total=num_examples):
        file_path_bytes = episode['episode_metadata']['file_path'].numpy()  # 返回 bytes
        file_path_str = file_path_bytes.decode('utf-8')                     # 转 str

        # get_language = next(iter(episode['steps'])) 
        # language = get_language['language_instruction'].numpy().decode()  
        # print('指令:', language)

        episode_path = Path(args.outpu_dir)/'episodes'/str(episode_index).zfill(6)  # 创建episodes文件夹
        os.makedirs(episode_path, exist_ok=True)   

        for steps_index,steps in enumerate(episode['steps']):  
            
            steps_path = episode_path/'steps'/str(steps_index).zfill(4)   # 创建steps文件夹
            os.makedirs(steps_path, exist_ok=True) 

            with h5py.File(f'{steps_path}/data.h5', 'w') as h5_file: # 创建 data.h5 文件
                image_primary = steps['observation']['image'].numpy()   
                image_wrist = steps['observation']['wrist_image'].numpy()
                language_instructions = steps['language_instruction'].numpy().decode()
                action = steps['action'].numpy()
                joint_state = steps['observation']['joint_state'].numpy()
                proprio_state = steps['observation']['state'].numpy()   # (8,) EEF_state', 'gripper_state'
                if steps_index == 0:
                    gripper_state = action[-1]  # (1,) gripper_state
                    old_gripper_state = gripper_state
                else:
                    gripper_state = old_gripper_state
                    old_gripper_state = action[-1]
                # language instruction
                h5_file.create_dataset('language_instruction', data=np.array(language_instructions, dtype=h5py.string_dtype(encoding='utf-8')))

                # episode length
                h5_file.create_dataset(name='episode_length', data=len(episode['steps']))

                # action
                h5_file.create_dataset(name='action', data=action)
                
                # dataset_name
                h5_file.create_dataset(name='dataset_name', data=np.array(args.dataset_name, dtype=h5py.string_dtype(encoding='utf-8')))

                # observation (timestep, proprio, image_XXX)
                observation_group = h5_file.create_group(name='observation')

                # image
                ### image_primary
                Image.fromarray(image_primary).resize((224, 224), Image.BILINEAR).save(f'{steps_path}/image_primary.jpg') 
                ### image_wrist
                Image.fromarray(image_wrist).resize((224, 224), Image.BILINEAR).save(f'{steps_path}/image_wrist.jpg')

                ## proprio
                observation_group.create_dataset(name='proprio', data=proprio_state)

                ## tcp_pose
                observation_group.create_dataset(name='tcp_pose', data=proprio_state[:6])

                ## gripper state (-1 or 1)
                observation_group.create_dataset(name='gripper_state', data=gripper_state)

                ## gripper position (n, 2)
                observation_group.create_dataset(name='gripper_position', data=proprio_state[-2:])
        
        with h5py.File(f'{episode_path}/step_info.h5', 'w') as h5_file:            # 创建step_info.h5
            h5_file.create_dataset(name='length', data=len(episode['steps']))
        
        episode_index += 1

    with h5py.File(f'{str(args.outpu_dir)}/episodes_info.h5', 'w') as h5_file:     # 创建episodes_info.h5
        episodes_dir = Path(args.outpu_dir)/'episodes'
        num_episodes = sum(1 for _ in episodes_dir.iterdir() if _.is_dir())
        h5_file.create_dataset(name='num_episodes', data=num_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/lxx/project/LIBERO-plus/datasets/rlds/libero_plus_data_4suite')   # 文件夹路径
    parser.add_argument('--suit_name', type=str, default='libero_goal_no_noops')                                                 # 数据集名称，libero_goal_no_noops | libero_object_no_noops
    parser.add_argument('--outpu_dir', type=str, default='/home/lxx/project/DreamVLA/datasets/Dream-adapter_datasets_goal_test') # 输出文件夹路径
    parser.add_argument('--dataset_name', type=str, default='libero_goal')                                                       # 数据集名称，libero_goal | libero_object
    parser.add_argument('--num_examples', type=int, default=5)  # 可选参数，指定要处理的episodes数量，None表示处理全部episodes
    args = parser.parse_args()
    main(args)
    print('finish!')
