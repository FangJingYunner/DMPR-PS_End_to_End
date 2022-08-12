"""Defines the parking slot dataset for directional marking point detection."""
import json
import os
import os.path
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from data.struct import MarkingPoint,ParkingSlot
import scipy.io as scio
import math
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch

class ParkingSlotDataset(Dataset):
    """Parking slot dataset."""
    def __init__(self, arg,is_train=True):
        super(ParkingSlotDataset, self).__init__()
        self.arg = arg
        
        if is_train == True:
            self.dataroot = arg.dataset_directory
            self.labelsroot = arg.labels_directory
        else:
            self.dataroot = arg.val_dataset_directory
            self.labelsroot = arg.val_labels_directory
            
        self.sample_names = []
        self.image_transform = ToTensor()
        # for file in os.listdir(root):
        #     if file.endswith(".json"):
        #         self.sample_names.append(os.path.splitext(file)[0])
        self.db = self._get_db()
        self.inputsize = 512
        self.ROT_FACTOR = 30
        self.TRANSLATE = 0.0
        self.SCALE_FACTOR = 0.0
        self.SHEAR = 0.0
        self.is_train = is_train
        self.HSV_H = 0.1
        self.HSV_S = 0.7
        self.HSV_V = 0.4
        self.lr_flip = True
        self.ud_flip = True

    def _get_db(self):
        gt_db = []
        label_list = os.listdir(self.labelsroot)
        no_slot_count = 0
        for label in tqdm.tqdm(label_list):
            
            if len(gt_db) > 20:
                return gt_db
            
            label_path = os.path.join(self.labelsroot,label)
            img_name = label.split(".")[0] + '.jpg'
            image_path = os.path.join(self.dataroot,img_name)
            with open(label_path, 'r') as file:
                jsonlabel = json.load(file)

                centralied_marks = np.array(jsonlabel['marks'])
                slots = np.array(jsonlabel['slots'])
                if len(slots) == 0:
                    no_slot_count += 1
                    continue

                if len(centralied_marks.shape) < 2:
                    centralied_marks = np.expand_dims(centralied_marks, axis=0)
                if len(slots.shape) < 2:
                    slots = np.expand_dims(slots, axis=0)
                    
                # centralied_marks[:, 0:4] -= 300.5
                generated_slot = []
                # if len(slots) == 4 and isinstance(slots[0],int):
                #     slots = [slots]
                
                for slot in slots:
                    mark0 = centralied_marks[slot[0]-1]
                    mark1 = centralied_marks[slot[1]-1]
                    mark0 = mark0 - 300.5
                    mark1 = mark1 - 300.5
                    mark0 = (mark0 + 300) / 600
                    mark1 = (mark1 + 300) / 600
                    
                    x1 = mark0[0]
                    y1 = mark0[1]

                    x2 = mark1[0]
                    y2 = mark1[1]
                    
                    direction = math.atan2(mark0[3] - mark0[1], mark0[2] - mark0[0])
                    generated_slot.append([x1,y1,x2,y2,direction])

            rec = [{
                'slot': jsonlabel,
                'image' : image_path

            }]

            gt_db += rec
        print("train data num: {}".format(len(gt_db)))
        print("not slot count: {}".format(no_slot_count))
        return gt_db

    def __getitem__(self, index):
        data = self.db[index]
        
        
        


        image = cv2.imread(data["image"])
        origin_height,origin_width,_= image.shape
        label = data["slot"]
        marking_points = []

        # image_path = "/media/fjy/SHARE/dataset/ps2.0/training/image20160722192751_1676.jpg"
        # image_path = "/media/fjy/SHARE/dataset/ps2.0/output_directory/train/image20160722192751_1676.jpg"
        # label_path = "/media/fjy/SHARE/dataset/ps2.0/ps_json_label/training/image20160722192751_1676.json"
        # with open(label_path, 'r') as file:
        #     label = json.load(file)
        # image = cv2.imread(image_path)
        
        if self.is_train:
        
            img, labels = self.random_perspective(
                combination=image,
                targets=label,
                degrees=self.ROT_FACTOR,
                translate=self.TRANSLATE,
                scale=self.SCALE_FACTOR,
                shear=self.SHEAR
            )
            
            img2 = self.augment_hsv(img, hgain=self.HSV_H, sgain=self.HSV_S, vgain=self.HSV_V)

            # imgshow = Image.fromarray(img)
            # imgshow.show()
            # img2show = Image.fromarray(img2)
            # img2show.show()
            img2 = np.ascontiguousarray(img2)
            
            img2 = self.image_transform(img2)
        else:
    
            image = cv2.resize(image, (self.inputsize, self.inputsize))
            img2 = np.ascontiguousarray(image)
            img2 = self.image_transform(img2)
            
            marks = np.array(label["marks"])
            slots = np.array(label["slots"])
    
            if len(marks.shape) < 2:
                marks = np.expand_dims(centralied_marks, axis=0)
            if len(slots.shape) < 2:
                slots = np.expand_dims(slots, axis=0)
                
            n = len(slots)
            labels = []
            if n:
                for slot in slots:
                    mark0 = marks[slot[0] - 1][:4]
                    mark1 = marks[slot[1] - 1][:4]
                    # mark0 = mark0 - 300.5
                    # mark1 = mark1 - 300.5
                    mark0 = (mark0 / origin_height) * self.inputsize
                    mark1 = (mark1 / origin_height) * self.inputsize
                    mark = np.vstack((mark0, mark1))
                    xy = np.ones((4, 3))
                    xy[:, :2] = mark.reshape(4, 2)
                    M = np.eye(3)
                    xy = xy @ M.T
                    # if perspective:
                    #     xy = (xy[:, :2] / xy[:, 2:3]).reshape(1, 8)  # rescale
                    # else:  # affine
                    xy = xy[:, :2].reshape(1, 8)
                    mark0 = xy[:, [0, 1, 2, 3]]
                    mark1 = xy[:, [4, 5, 6, 7]]
            
                    mark0 = mark0 - self.inputsize / 2 + 0.5
                    mark1 = mark1 - self.inputsize / 2 + 0.5
                    mark0 = (mark0 + self.inputsize / 2) / self.inputsize
                    mark1 = (mark1 + self.inputsize / 2) / self.inputsize
            
                    # if use_ud_filp:
                    #     mark0[:, [1, 3]] = 1 - mark0[:, [1, 3]]
                    #     mark1[:, [1, 3]] = 1 - mark1[:, [1, 3]]
                    # if use_lr_flip:
                    #     mark0[:, [0, 2]] = 1 - mark0[:, [0, 2]]
                    #     mark1[:, [0, 2]] = 1 - mark1[:, [0, 2]]
            
                    x1 = mark0[:, 0]
                    y1 = mark0[:, 1]
            
                    x2 = mark1[:, 0]
                    y2 = mark1[:, 1]
            
                    direction = math.atan2(mark0[:, 3] - mark0[:, 1], mark0[:, 2] - mark0[:, 0])
                    labels.append(ParkingSlot(*[x1, y1, x2 - x1, y2 - y1, direction]))

        # labels = torch.from_numpy(labels)
        # name = self.sample_names[index]
        # image = cv.imread(os.path.join(self.root, name+'.jpg'))
        # image = self.image_transform(image)
        # with open(os.path.join(self.root, name + '.json'), 'r') as file:
        #     for label in json.load(file):
        #         marking_points.append(MarkingPoint(*label))
        return img2, labels

    def __len__(self):
        return len(self.db)

    def augment_hsv(self,img, hgain=0.5, sgain=0.5, vgain=0.5):
        """change color hue, saturation, value"""
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8
    
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        # cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img2
        # Histogram equalization
        # if random.random() < 0.2:
        #     for i in range(3):
        #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])

    def random_perspective(self,combination, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                           border=(0, 0)):
        """combination of img transform"""
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [cls, xyxy]
        img = combination
        origin_height,origin_width,_= img.shape

        img = cv2.resize(img, (self.inputsize, self.inputsize))

        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        not_good_augmentation = True
        re_augmentation = 1
        while(not_good_augmentation):
            not_good_augmentation = False
            # Center
            C = np.eye(3)
            C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
            C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
        
            # Perspective
            P = np.eye(3)
            P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
            P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
        
            # Rotation and Scale
            R = np.eye(3)
            a = random.uniform(-degrees, degrees)
            # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = random.uniform(1 - scale, 1 + scale)
            # s = 2 ** random.uniform(-scale, scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        
            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
        
            # Translation
            T = np.eye(3)
            T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
            T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
    
            marks = np.array(targets["marks"])
            slots = np.array(targets["slots"])
            
            if len(marks.shape) < 2:
                marks = np.expand_dims(marks, axis=0)
            if len(slots.shape) < 2:
                slots = np.expand_dims(slots, axis=0)
            
            # Combined rotation matrix
            M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
            
            # for slot in slots[:,:2]:
            #     for mark_id in slot:
            #         mark = marks[mark_id - 1]
            #         x = mark[0]
            #         y = mark[1]
            #         if x < 100 or x > 500 or y < 100 or y > 500:
            #             M = np.eye(3)
    
            
            
            if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
                if perspective:
                    img2 = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(0, 0, 0))
                    # gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
                    # line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
                else:  # affine
                    img2 = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
                    # gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)
                    # line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)
            else:
                img2 = img.copy()
            # Visualize
            # ax = plt.subplots(1, 2, figsize=(18, 9))[1].ravel()
            # ax[0].imshow(img[:, :, ::-1])  # base
            # ax[1].imshow(img2[:, :, ::-1])  # warped
            # plt.show()
            # Transform label coordinates
    
            use_lr_flip = False
            if self.lr_flip and random.random() < 0.85:
                use_lr_flip = True
                img2 = np.fliplr(img2)
            use_ud_filp = False
            if self.ud_flip and random.random() < 0.85:
                use_ud_filp = True
                img2 = np.flipud(img2)
                
            # if lr_flip and random.random() < 0.5:
            #
            #     mark0[:, [0, 2]] = 1 - mark0[:, [0, 2]]
            #     mark1[:, [0, 2]] = 1 - mark1[:, [0, 2]]
            #
            # # random up-down flip
            # ud_flip = True
            # # if ud_flip and random.random() < 0.5:
            # if ud_flip:
            #     img2 = np.flipud(img2)
            #     mark0[:, [1, 3]] = 1 - mark0[:, [1, 3]]
            #     mark1[:, [1, 3]] = 1 - mark1[:, [1, 3]]
            #     # mark0= 1 - mark0
            #     # mark1 = 1 - mark0
    
            
            n = len(slots)
            generated_slot = []
            if n:
                for slot in slots:
                    mark0 = marks[slot[0] - 1][:4]
                    mark1 = marks[slot[1] - 1][:4]
                    # mark0 = mark0 - 300.5
                    # mark1 = mark1 - 300.5
                    mark0 = (mark0  / origin_height)*self.inputsize
                    mark1 = (mark1  / origin_height)*self.inputsize
                    mark = np.vstack((mark0,mark1))
                    xy = np.ones((4, 3))
                    xy[:, :2] = mark.reshape(4, 2)
                    
                    # M = np.eye(3)
                    xy = xy @ M.T
    
                        
                    if perspective:
                        xy = (xy[:, :2] / xy[:, 2:3]).reshape(1, 8)  # rescale
                    else:  # affine
                        xy = xy[:, :2].reshape(1, 8)
                    mark0 = xy[:, [0, 1, 2, 3]]
                    mark1 = xy[:, [4, 5, 6, 7]]
    
                    mark0 = mark0 - self.inputsize/2 + 0.5
                    mark1 = mark1 - self.inputsize/2 + 0.5
                    mark0 = (mark0 + self.inputsize/2) / self.inputsize
                    mark1 = (mark1 + self.inputsize/2) / self.inputsize
    
                    if use_ud_filp:
                        mark0[:, [1, 3]] = 1 - mark0[:, [1, 3]]
                        mark1[:, [1, 3]] = 1 - mark1[:, [1, 3]]
                    if use_lr_flip:
                        mark0[:, [0, 2]] = 1 - mark0[:, [0, 2]]
                        mark1[:, [0, 2]] = 1 - mark1[:, [0, 2]]
                        
                    x1 = mark0[:,0]
                    y1 = mark0[:,1]
                    x2 = mark1[:,0]
                    y2 = mark1[:,1]
    
                    # x1 = mark0[0]
                    # y1 = mark0[1]
                    # x2 = mark1[0]
                    # y2 = mark1[1]
                    
                    direction = math.atan2(mark0[:,3] - mark0[:,1], mark0[:,2] - mark0[:,0])
                    generated_slot.append(ParkingSlot(*[float(x1), float(y1), float(x2-x1), float(y2-y1), float(direction)]))

                    #img2show = Image.fromarray(img2)
                    #imgshow = Image.fromarray(img)
                    #imgdraw = ImageDraw.Draw(img2show)
                    #imgdraw.line((int(mark0[:, 0] * self.inputsize), int(mark0[:, 1] * self.inputsize),
                    #              int(mark0[:, 2] * self.inputsize), int(mark0[:, 3] * self.inputsize)), fill='yellow',
                    #             width=3)
                    #imgdraw.line((int(mark1[:, 0] * self.inputsize), int(mark1[:, 1] * self.inputsize),
                    #              int(mark1[:, 2] * self.inputsize), int(mark1[:, 3] * self.inputsize)), fill='yellow',
                    #             width=3)
                    #img2show.show()
                    # imgshow.show()
                
                    for slot in generated_slot:
                        x,y = slot.x,slot.y
                        # print(x,y)
                        if x < 0 or y < 0 or x > 1 or y > 1:
                            # print("re_augmentation:{}".format(re_augmentation))
                            re_augmentation += 1
                            not_good_augmentation = True
                            break
                        
                

                    
                    
                    
                #     ax = plt.subplots(1, 2, figsize=(18, 9))[1].ravel()
                #     ax[0].imshow(img[:, :, ::-1])  # base
                #     cv2.line(img2, (int(mark0[:,0]*self.inputsize), int(mark0[:,1]*self.inputsize)), (int(mark0[:,2]*self.inputsize), int(mark0[:,3]*self.inputsize)), (153, 50, 204), 10)
                #     cv2.line(img2, (int(mark1[:,0]*self.inputsize), int(mark1[:,1]*self.inputsize)), (int(mark1[:,2]*self.inputsize), int(mark1[:,3]*self.inputsize)), (153, 50, 204), 10)
                #     ax[1].imshow(img2[:, :, ::-1])  # warped
                # plt.show()

        return img2,generated_slot
        
        # if n:
        #     # warp points
        #     xy = np.ones((n * 4, 3))
        #     xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        #     xy = xy @ M.T  # transform
        #     if perspective:
        #         xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        #     else:  # affine
        #         xy = xy[:, :2].reshape(n, 8)
        #
        #     # create new boxes
        #     x = xy[:, [0, 2, 4, 6]]
        #     y = xy[:, [1, 3, 5, 7]]
        #     xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        #
        #     # # apply angle-based reduction of bounding boxes
        #     # radians = a * math.pi / 180
        #     # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        #     # x = (xy[:, 2] + xy[:, 0]) / 2
        #     # y = (xy[:, 3] + xy[:, 1]) / 2
        #     # w = (xy[:, 2] - xy[:, 0]) * reduction
        #     # h = (xy[:, 3] - xy[:, 1]) * reduction
        #     # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
        #
        #     # clip boxes
        #     xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        #     xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        #
        #     # filter candidates
        #     i = _box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        #     targets = targets[i]
        #     targets[:, 1:5] = xy[i]
        #
        # combination = (img, gray, line)
        # return combination, targets