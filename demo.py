import aug as am
import Helpers as hp
from util import *
import os
from os.path import join
from tqdm import tqdm
import random
import shutil

# base_dir = os.getcwd()#得到当前程序所在的目录
#
# save_base_dir = join(base_dir, 'save')

# check_dir(save_base_dir)

# imgs_dir = [f.strip() for f in open(join(base_dir, 'train.txt')).readlines()]
# labels_dir = hp.replace_labels(imgs_dir)
#
# small_imgs_dir = [f.strip() for f in open(join(base_dir, 'small.txt')).readlines()]
# random.shuffle(small_imgs_dir)
#
#
# for image_dir, label_dir in tqdm(zip(imgs_dir, labels_dir)):
#     # small_img = []
#     # for x in range(8):
#     #     if small_imgs_dir == []:
#     #         #exit()
#     #         small_imgs_dir = [f.strip() for f in open(join(base_dir, 'small.txt')).readlines()]
#     #         random.shuffle(small_imgs_dir)
#     #     small_img.append(small_imgs_dir.pop())
#     am.copysmallobjects2(image_dir, label_dir, save_base_dir)

image_dir = 'D:\\ustc\\SmallObjectAugmentation-masterv2\\dotatest'
label_dir = 'D:\\ustc\\SmallObjectAugmentation-masterv2\\dotatestlabel'
image_ag_dir = 'D:\\ustc\\SmallObjectAugmentation-masterv2\\data_ag'
label_ag = 'D:\\ustc\\SmallObjectAugmentation-masterv2\\label_ag'

error_img = 'D:\\ustc\\SmallObjectAugmentation-masterv2\\error_img'
error_label = 'D:\\ustc\\SmallObjectAugmentation-masterv2\\error_label'

if os.path.exists(error_img):
    shutil.rmtree(error_img)
os.makedirs(error_img)

if os.path.exists(error_label):
    shutil.rmtree(error_label)
os.makedirs(error_label)

if os.path.exists(image_ag_dir):
    shutil.rmtree(image_ag_dir)
os.makedirs(image_ag_dir)
if os.path.exists(label_ag):
    shutil.rmtree(label_ag)
os.makedirs(label_ag)

k = 0
while k<=0:

    i = 0
    lists = os.listdir(image_dir)
    for image in lists:
        abspath1 = join(image_dir, image)
        abspath2 = join(label_dir, image.replace('.jpg', '.txt'))
        try:
            am.copysmallobjects2(abspath1, abspath2,image_ag_dir,label_ag,i+k*len(lists))
            print('the '+str(k)+' epoch: '+'finished '+str(i+1)+' pictures')
            i+=1
        except:
            shutil.copy(abspath1,error_img)
            shutil.copy(abspath2,error_label)
    k+=1

# for img_dir,label_dir in tqdm(zip(imgs_dir,labels_dir)):
#     img = cv2.imread(img_dir)
#     labels =