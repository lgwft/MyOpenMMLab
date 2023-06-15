import numpy as np
import time
import shutil

import torch

from PIL import Image
import cv2
import os

import mmcv
import mmengine
from mmseg.apis import inference_model
from mmseg.utils import register_all_modules
register_all_modules()

from mmseg.datasets import CityscapesDataset

# 模型 config 配置文件
config_file = 'E:\github\MyOpenMMLab\homework_4_mmseg\configs\pspnet-DubaiDataset_20230612.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'E:\github\MyOpenMMLab\homework_4_mmseg\workdir\iter_3000.pth'

from mmseg.apis import init_model
model = init_model(config_file, checkpoint_file, device='cuda:0')

from mmengine.model.utils import revert_sync_batchnorm
if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)

# input_video = 'data/traffic.mp4'

input_video = 'E:\github\MyOpenMMLab\homework_4_mmseg\dataset\\test\\test.mp4'

temp_out_dir = time.strftime('%Y%m%d%H%M%S')
os.mkdir(temp_out_dir)
print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
# 读入待预测视频
imgs = mmcv.VideoReader(input_video)

prog_bar = mmengine.ProgressBar(len(imgs))
from mmseg.datasets import cityscapes
classes = cityscapes.CityscapesDataset.METAINFO['classes']
palette = cityscapes.CityscapesDataset.METAINFO['palette']


def pridict_single_frame(img, opacity=0.2):
    result = inference_model(model, img)

    # 将分割图按调色板染色
    seg_map = np.array(result.pred_sem_seg.data[0].detach().cpu().numpy()).astype('uint8')
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))

    show_img = (np.array(seg_img.convert('RGB'))) * (1 - opacity) + img * opacity

    return show_img

# 对视频逐帧处理
for frame_id, img in enumerate(imgs):
    ## 处理单帧画面
    show_img = pridict_single_frame(img, opacity=0.15)
    temp_path = f'{temp_out_dir}/{frame_id:06d}.jpg'  # 保存语义分割预测结果图像至临时文件夹
    cv2.imwrite(temp_path, show_img)

    prog_bar.update()  # 更新进度条

# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, 'E:\github\MyOpenMMLab\homework_4_mmseg\\results\\test.mp4', fps=imgs.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)