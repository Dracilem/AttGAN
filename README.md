Implementation DMT1909215

Commands
Load the project:
!git clone https://github.com/Dracilem/AttGAN

import os
os.chdir('/content/AttGAN/')

!pip install -r requirements.txt

Train:

!python train.py --dataroot ./datasets/Mydatasets --name ModelName --vgg --self_attention

Test:

!python test.py --dataroot datasets/Mydatasets/testA --name ModelName --self_attention --self_attention_thresh 0.8 --no_dropout

Evaluate:

import os
os.chdir('/content/AttGAN/evaluation')
!python evaluate.py --org_img_path D:/Github/AttGAN/evaluation/input/ --pred_img_path D:/Github/AttGAN/evaluation/Improved40 --metric psnr
(--org_img_path is the absolute path of original images, --pred_img_path is the absolute path of improved images)