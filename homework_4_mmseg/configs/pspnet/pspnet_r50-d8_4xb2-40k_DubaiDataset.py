_base_ = [
    '../../mmsegmentation/configs/_base_/models/pspnet_r50-d8.py', 'E:\github\MyOpenMMLab\homework_4_mmseg\mmsegmentation\configs\_base_\datasets\DubaiDataset_pipeline.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py', '../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]
crop_size = (64, 64) # 输入图像尺寸，根据自己数据集情况修改
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
