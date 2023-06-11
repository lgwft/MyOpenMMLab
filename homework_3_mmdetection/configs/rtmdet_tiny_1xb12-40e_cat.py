
_base_ = 'E:\github\MyOpenMMLab\homework_3_mmdetection\mmdetection\configs\\rtmdet\\rtmdet_tiny_8xb32-300e_coco.py'

data_root = 'E:\github\MyOpenMMLab\homework_3_mmdetection\data\Drink_284_Detection_coco'


metainfo = {
    'classes': ('cola', 'pepsi', 'sprite', 'fanta', 'spring', 'ice', 'scream', 'milk', 'red', 'king'),
    'palette': [
        (255, 0, 0),    
        (0, 255, 0),    
        (0, 0, 255),    
        (255, 255, 0),  
        (255, 165, 0), 
        (128, 0, 128),  
        (255, 192, 203),  
        (255, 255, 255),  
        (255, 0, 0),    
        (0, 0, 0),     
    ]
}

num_classes = 10


max_epochs = 40

train_batch_size_per_gpu = 12

train_num_workers = 4


val_batch_size_per_gpu = 1
val_num_workers = 2


num_epochs_stage2 = 5


base_lr = 12 * 0.004 / (32*8)


load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'  
model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    pin_memory=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='E:\github\MyOpenMMLab\homework_3_mmdetection\data\Drink_284_Detection_coco\\train_coco.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='E:\github\MyOpenMMLab\homework_3_mmdetection\data\Drink_284_Detection_coco\\val_coco.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=30),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,  
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
optim_wrapper = dict(optimizer=dict(lr=base_lr))


_base_.custom_hooks[1].switch_epoch = max_epochs - num_epochs_stage2

val_evaluator = dict(ann_file=data_root + '\\val_coco.json')
test_evaluator = val_evaluator

# һЩ��ӡ�����޸�
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'), 
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
