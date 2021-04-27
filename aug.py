# import glob
import cv2 as cv2
import numpy as np
# from PIL import Image
import random
import math
from tqdm import tqdm
from os.path import basename, split, join, dirname
from util import *


def find_str(filename):
    if 'train' in filename:
        return dirname(filename[filename.find('train'):])
    else:
        return dirname(filename[filename.find('val'):])


def convert_all_boxes(shape, anno_infos, yolo_label_txt_dir):
    height, width, n = shape
    label_file = open(yolo_label_txt_dir, 'w')
    for anno_info in anno_infos:
        target_id, x1, y1, x2, y2 = anno_info
        b = (float(x1), float(x2), float(y1), float(y2))
        bb = convert((width, height), b)  # 将边界框归一化，得到yolo格式的数据列表
        label_file.write(str(target_id) + " " + " ".join([str(a) for a in bb]) + '\n')  # 将列表写入到txt文件中；（注意写入的方法!）


def save_crop_image(save_crop_base_dir, image_dir, idx, roi):
    crop_save_dir = join(save_crop_base_dir, find_str(image_dir))
    check_dir(crop_save_dir)
    crop_img_save_dir = join(crop_save_dir, basename(image_dir)[:-3] + '_crop_' + str(idx) + '.jpg')
    cv2.imwrite(crop_img_save_dir, roi)


def copysmallobjects(image_dir, label_dir, save_base_dir, save_crop_base_dir=None,
                     save_annoation_base_dir=None):
    image = cv2.imread(image_dir)

    labels = read_label_txt(label_dir)
    if len(labels) == 0: return
    rescale_labels = rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    all_boxes = []

    for idx, rescale_label in enumerate(rescale_labels):

        all_boxes.append(rescale_label)
        # 目标的长宽
        rescale_label_height, rescale_label_width = rescale_label[4] - rescale_label[2], rescale_label[3] - \
                                                    rescale_label[1]

        if (issmallobject((rescale_label_height, rescale_label_width), thresh=64 * 64) and rescale_label[0] == '1'):
            roi = image[rescale_label[2]:rescale_label[4], rescale_label[1]:rescale_label[3]]

            new_bboxes = random_add_patches(rescale_label, rescale_labels, image.shape, paste_number=2, iou_thresh=0.2)
            count = 0

            # 将新生成的位置加入到label,并在相应位置画出物体
            for new_bbox in new_bboxes:
                count += 1
                all_boxes.append(new_bbox)
                cl, bbox_left, bbox_top, bbox_right, bbox_bottom = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], \
                                                                   new_bbox[4]
                try:
                    if (count > 1):
                        roi = flip_bbox(roi)
                    image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                except ValueError:
                    continue

    dir_name = find_str(image_dir)
    save_dir = join(save_base_dir, dir_name)
    check_dir(save_dir)
    yolo_txt_dir = join(save_dir, basename(image_dir.replace('.jpg', '_augment.txt')))
    cv2.imwrite(join(save_dir, basename(image_dir).replace('.jpg', '_augment.jpg')), image)
    convert_all_boxes(image.shape, all_boxes, yolo_txt_dir)


def GaussianBlurImg(image):
    # 高斯模糊
    ran = random.randint(0, 9)
    if ran % 2 == 1:
        image = cv2.GaussianBlur(image, ksize=(ran, ran), sigmaX=0, sigmaY=0)
    else:
        pass
    return image


def suo_fang(image, area_max=2000, area_min=1000):
    # 改变要粘贴的图片的大小，面积大小值存在于1000~2000之间
    height, width, channels = image.shape

    while (height*width) > area_max:
        image = cv2.resize(image, (int(width * 0.9),int(height * 0.9)))
        height, width, channels = image.shape
        height,width = int(height*0.9),int(width*0.9)

    while (height*width) < area_min:
        image = cv2.resize(image, (int(15),int(15)))
        height, width, channels = image.shape
        height, width = int(15),int(15)

    return image


def copysmallobjects2(image_dir, label_dir, img_save_dir, label_save_dir,k):
    image = cv2.imread(image_dir)
    labels = read_label_txt(label_dir)
    # labels = label_dir
    if len(labels) == 0:
        return
    rescale_labels = rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    all_boxes = []
    small_object = []
    small_label = []
    for i, rescale_label in enumerate(rescale_labels):
        #cv2.rectangle(image, (rescale_label[1], rescale_label[2]), (rescale_label[3], rescale_label[4]), (0, 0, 255), 2)
        #cv2.imshow('src', image)
        #cv2.waitKey()
        dst = image[rescale_label[2]:rescale_label[4], rescale_label[1]:rescale_label[3]]  # 裁剪坐标为[y0:y1 , x0:x1]
        #cv2.imshow('image', dst)
        #cv2.waitKey()
        #roi = suo_fang(dst, area_max=3000, area_min=1500)
        if i <= 10:
            small_object.append(dst)
            small_label.append(int(rescale_label[0]))
        all_boxes.append(rescale_label)

    #for small_img_dirs in small_img_dir:
    for image_bbox in small_object:
        #image_bbox = cv2.imread(small_img_dirs)
        image_label = small_label[small_object.index(image_bbox)]
        
        if image_label == 1:
            roi = suo_fang(image_bbox, area_max=5000, area_min=225)  # ROI就指要粘贴的每一个图片(因为是在一个for循环中)(尺寸大小是修正后的)
            new_bboxes = random_add_patches2(roi.shape, rescale_labels, image_label, image.shape, iou_thresh=0)
            count = 0
            for new_bbox in new_bboxes:
                count += 1

                cl, bbox_left, bbox_top, bbox_right, bbox_bottom = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], \
                                                                new_bbox[4]
                #roi = GaussianBlurImg(roi)  # 高斯模糊
                height, width, channels = roi.shape
                center = (int(width / 2),int(height / 2))
                #ran_point = (int((bbox_top+bbox_bottom)/2),int((bbox_left+bbox_right)/2))
                mask = 255 * np.ones(roi.shape, roi.dtype)

                try:
                    if count > 1:
                        roi = flip_bbox(roi) # 水平翻转要重复粘贴的图片！
                    #image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                    #image[bbox_top:bbox_bottom, bbox_left:bbox_right] = cv2.addWeighted(image[bbox_top:bbox_bottom, bbox_left:bbox_right],
                    #                                                                    0.5,roi,0.5,0) #图片融合

                    # 泊松融合
                    #image = cv2.seamlessClone(roi, image, mask, ran_point, cv2.NORMAL_CLONE)
                    #print(str(bbox_bottom-bbox_top) + "|" + str(bbox_right-bbox_left))
                    #print(roi.shape)
                    #print(mask.shape)
                    image[bbox_top:bbox_bottom, bbox_left:bbox_right] = cv2.seamlessClone(roi, image[bbox_top:bbox_bottom, bbox_left:bbox_right],
                                                                                        mask, center, cv2.NORMAL_CLONE)
                    
                    all_boxes.append(new_bbox)
                    rescale_labels.append(new_bbox)  # 保证了每一个新加入的new_bbox都被添加到了原始图像中，这样可以防止粘贴的图像和之前粘贴的图像发生重叠！
                except ValueError:
                    print("---")
                    continue
        else:
            pass
    # dir_name = find_str(image_dir)
    #img_save_dir = 
    # check_dir(save_dir)
    #label_save_dir = split(label_dir)[0]
    yolo_txt_dir = join(label_save_dir, basename(image_dir.replace('.jpg', '_augment'+str(k)+'.txt')))
    cv2.imwrite(join(img_save_dir, basename(image_dir).replace('.jpg', '_augment'+str(k)+'.jpg')), image)  # 生成添加小目标的jpg图片
    convert_all_boxes(image.shape, all_boxes, yolo_txt_dir)  # 生成添加小目标的txt标签(yolo格式)
