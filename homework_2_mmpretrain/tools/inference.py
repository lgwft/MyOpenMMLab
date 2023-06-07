from mmpretrain import ImageClassificationInferencer

inferencer = ImageClassificationInferencer('E:\github\MyOpenMMLab\homework_2_mmpretrain\configs\\resnet50_fruits30.py',pretrained='E:\github\MyOpenMMLab\homework_2_mmpretrain\work_dir_pretrain\epoch_17.pth')

inferencer("E:\github\MyOpenMMLab\homework_2_mmpretrain\data\\test.jpg",show_dir='E:\github\MyOpenMMLab\homework_2_mmpretrain\\result')