from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import random
from PIL import Image
from operator import itemgetter, attrgetter

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import time
from utils.utils import AverageMeter

flexible_list = ["KAERI_bandage", "KAERI_clothes", "KAERI_cotton", "KAERI_decontamination_paper", "KAERI_sandpaper",
                 "KAERI_shoe_net", "KAERI_tissue", "KAERI_vinyl_gloves", "KAERI_rubber_glove", "KAERI_ranch_glove"]

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def get_rotated_info(w, h, center, M):
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]
    return (bound_w, bound_h), M


def rotate_points(p, M):
    points = []
    for pt in p:
        v = [pt[0], pt[1], 1]
        calculated = np.dot(M, v)
        points.append((int(calculated[0]), int(calculated[1])))
    return points


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# points=rotate_points([(0,0),(pw,0),(0,ph),(pw,ph)],M)
# for i  in range(len(anns)):
# ann = anns[i]
# img=cv2.rectangle(img,(ann[0],ann[1]),(ann[2],ann[3]),(0,0,255),8)
# img=cv2.putText(img, ann[4], (ann[0],ann[1]-20), cv2.FONT_HERSHEY_COMPLEX, 3.5, (0,0,255), 2)
# cv2.imshow("img",cv2.resize(img,(512,512)))
# cv2.waitKey(0)	


class CenternetDetPatchSampler(data.Dataset):
    def get_offset(self, xmax, ymax, pw, ph, anns):
        def gen():
            offset_x = 0 if (int(xmax) == 0) else np.random.randint(0, xmax, size=1)[0]
            offset_y = 0 if (int(ymax) == 0) else np.random.randint(0, ymax, size=1)[0]
            return offset_x, offset_y

        offset_x = 0
        offset_y = 0
        if (len(anns) == 0):
            offset_x, offset_y = gen()
        else:
            boxes = np.array(anns)
            boxes = boxes[:, :4]
            coord = boxes.reshape(-1, 2).astype(np.int)
            minx = np.min(coord[:, 0])
            miny = np.min(coord[:, 1])
            maxx = np.max(coord[:, 0])
            maxy = np.max(coord[:, 1])
            x_candidate = list(range(0, minx)) + list(range(maxx, xmax + pw))
            x_candidate = [_x for _x in x_candidate if _x <= xmax]
            y_candidate = list(range(0, miny)) + list(range(maxy, ymax + ph))
            y_candidate = [_y for _y in y_candidate if _y <= ymax]
            candidates = []
            initial_candiate_count = len(x_candidate) * len(y_candidate)
            while (True):
                x_count = len(x_candidate)
                y_count = len(y_candidate)

                x_idx = 0
                if (x_count == 0):
                    offset_x = 0 if (int(xmax) == 0) else np.random.randint(0, xmax, size=1)[0]
                else:
                    x_idx = np.random.randint(0, x_count, size=1)[0]
                    offset_x = x_candidate[x_idx]

                y_idx = 0
                if (y_count == 0):
                    offset_y = 0 if (int(ymax) == 0) else np.random.randint(0, ymax, size=1)[0]
                else:
                    y_idx = np.random.randint(0, y_count, size=1)[0]
                    offset_y = y_candidate[y_idx]

                bbox = [offset_x, offset_y, offset_x + pw, offset_y + ph]
                passed = True
                iou_sum = 0
                is_overlapped = False
                if (np.random.random() < 0.5):
                    is_overlapped = True
                for ann in anns:
                    ann_box = ann[:4]
                    iou = get_iou(bbox, ann_box)
                    iou_sum += iou
                    if (is_overlapped == True):
                        if (iou < 0.5):  # overlapped (iou < 0.5 not passed) 최대한 기준 만큼 겹처 치는 샘플만 나올수있도록
                            passed = False
                    else:
                        if (iou > 0.5):  # not overlapped (iou > 0.5 not passed) 최대한 기준 만큼 겹쳐지지 않는 샘플만 나올수있도록
                            passed = False
                if (passed == True):
                    break
                else:
                    if (x_count > 0):
                        x_candidate.pop(x_idx)
                    if (y_count > 0):
                        y_candidate.pop(y_idx)
                    candidates.append((bbox, iou_sum))
                    candidate_count = 100
                    if candidate_count > initial_candiate_count:
                        candidate_count = initial_candiate_count
                    if (len(candidates) > candidate_count):
                        def get_candicate_position(candidates, reverse=False):
                            sort_candidates = sorted(candidates, key=itemgetter(1), reverse=reverse)
                            offset_x = sort_candidates[0][0][0]
                            offset_y = sort_candidates[0][0][1]
                            return offset_x, offset_y

                        return get_candicate_position(candidates, is_overlapped)

        return offset_x, offset_y

    def pacth_to_data(self, max_objs=5):

        back_id= np.random.choice(list(range(0, len(self.back_labels))), 1)[0]
        choice_background =self.back_labels[back_id].copy()

        choice_patch_ids = np.random.choice(list(range(0, len(self.patch_labels))), max_objs)
        patch_dict = {}
        for id in choice_patch_ids:
            label = self.patch_labels[id]
            key = list(label.keys())[0]
            patch_idx = np.random.choice(list(range(0, len(label[key]))), 1)[0]
            patch = label[key][patch_idx][0]
            patch_dir=label[key][patch_idx][1]
            patch_dict[patch_dir] = (key,patch)

        ret_img =choice_background

        if (ret_img is None):
            return None, []

        # if np.random.random() < 0.5:
        #     ret_img= (ret_img.astype(np.float32) / 255.)
        #     color_aug(self._data_rng, ret_img, self._eig_val, self._eig_vec)
        #     ret_img=(ret_img*255).astype(np.uint8)

        patch_img_list = []
        for id in patch_dict.keys():
            patch_img = patch_dict[id][1].copy()
            if (patch_img is None):
                continue
            ph, pw, pc = patch_img.shape
            if np.random.random() < 0.5:
                screw_scale = 0.4
                pts_base = [[0, 0], [pw, 0], [0, ph], [pw, ph]]

                tl = pts_base[0].copy()
                tl[0] = np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
                tl[1] = np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]

                tr = pts_base[1].copy()
                tr[0] = tr[0] - np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
                tr[1] = np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]

                bl = pts_base[2].copy()
                bl[0] = np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
                bl[1] = bl[1] - np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]

                br = pts_base[3].copy()
                br[0] = br[0] - np.random.uniform(low=0, high=pw * screw_scale, size=1)[0]
                br[1] = br[1] - np.random.uniform(low=0, high=ph * screw_scale, size=1)[0]
                pts1 = np.float32([tl, tr, bl, br])

                pts_base = np.float32(pts_base)
                PM = cv2.getPerspectiveTransform(pts_base, pts1)
                patch_img = cv2.warpPerspective(patch_img, PM, (pw, ph))
                coords = np.argwhere(patch_img[:, :, 3] > 0)
                if (coords is None):
                    continue
                if (coords.shape[0] < 4):
                    continue
                x0, y0 = coords.min(axis=0)
                x1, y1 = coords.max(axis=0) + 1
                patch_img = patch_img[x0:x1, y0:y1]

            # if patch_dict[patch_dir] in flexible_list and np.random.random() < 0.5:
            #     h,w=  patch_img.shape[:2]
            #     size=h if(h<w) else w
            #     patch_img = elastic_transform(patch_img,size*3, size*0.07, size*0.09)
            #     coords = np.argwhere(patch_img[:, :, 3] > 0)
            #     x0, y0 = coords.min(axis=0)
            #     x1, y1 = coords.max(axis=0) + 1
            #     patch_img = patch_img[x0:x1, y0:y1]
            if np.random.random() < 0.5:
                patch_img= (patch_img.astype(np.float32) / 255.)
                color_aug(self._data_rng, patch_img, self._eig_val, self._eig_vec)
                patch_img=(patch_img*255).astype(np.uint8)

            ph, pw, pc = patch_img.shape

            if np.random.random() < 0.5:
                patch_img = patch_img[:, ::-1, :]
            if np.random.random() < 0.5:
                patch_img = patch_img[::-1, :, :]

            scale = np.random.uniform(low=0.1, high=0.9, size=1)[0]
            rotaion = np.random.choice([0, 90, 180, 270], 1)[0]

            image_center = (pw / 2, ph / 2)
            M = cv2.getRotationMatrix2D((image_center[0], image_center[1]), angle=rotaion, scale=scale)
            size, M = get_rotated_info(w=pw, h=ph, center=image_center, M=M)
            patch_img = cv2.warpAffine(patch_img, M, (size[0], size[1]))
            ph, pw, pc = patch_img.shape
            patch_img_list.append((patch_img, int(pw * ph), patch_dict[id][0]))

        patch_img_list = sorted(patch_img_list, key=itemgetter(1), reverse=True)

        bh, bw, bc = ret_img.shape
        anns = []
        for patch_img, _, category in patch_img_list:
            if (patch_img is None):
                continue
            ph, pw, pc = patch_img.shape
            resize = False
            sfx = sfy = ph
            if (ph > bh):
                sfy = bh / ph
                resize = True
            if (pw > bw):
                sfx = bw / pw
                resize = True

            if (resize == True):
                sf = sfx if sfx < sfy else sfy
                sf = sf * 0.5
                patch_img = cv2.resize(patch_img, dsize=None, fx=sf, fy=sf)

            patch_img_rgb = patch_img[:, :, :3]
            patch_img_mask = patch_img[:, :, 3]

            ph, pw, pc = patch_img.shape
            xmax = bw - pw
            ymax = bh - ph
            # offset_x, offset_y = self.get_offset(xmax, ymax, pw, ph, anns)
            offset_x=0
            if(xmax>0):
                offset_x=np.random.randint(0, xmax, size=1)[0]
            offset_y=0
            if(ymax>0):
                offset_y=np.random.randint(0, ymax, size=1)[0]

            fg = cv2.bitwise_and(patch_img_rgb, patch_img_rgb, mask=patch_img_mask)
            patch_img_mask = cv2.bitwise_not(patch_img_mask)
            bg = ret_img[offset_y:offset_y + ph, offset_x:offset_x + pw]
            bg = cv2.bitwise_or(bg, bg, mask=patch_img_mask)
            ret_img[offset_y:offset_y + ph, offset_x:offset_x + pw] = cv2.bitwise_or(fg, bg)
            anns.append([offset_x, offset_y, offset_x + pw, offset_y + ph, category])

        return ret_img, anns

    def negative_sampling(self):
        idx=np.random.choice(list(range(0, len(self.negative_labels))), 1)[0]
        negative_sample =self.negative_labels[idx].copy()
        return negative_sample

    def _label_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[2], box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        ret = []
        num_objs = 5
        img = None
        anns = []
        #end = time.time()
        if np.random.random() < 0.5 or self.negative_labels is None:
            while ((len(anns) > 0 and img is not None) != True):
                img, anns = self.pacth_to_data(max_objs=num_objs)
        else:
            img = self.negative_sampling()
        #print("\ngen time: {}".format(time.time() - end))

        # for i  in range(len(anns)):
        # 	ann = anns[i]
        # 	img=cv2.rectangle(img,(ann[0],ann[1]),(ann[2],ann[3]),(0,0,255),8)
        # 	img=cv2.putText(img, ann[4], (ann[0],ann[1]-20), cv2.FONT_HERSHEY_COMPLEX, 3.5, (0,0,255), 2)
        # cv2.imshow("img",cv2.resize(img,(512,512)))
        # cv2.waitKey(0)

        num_objs = min(len(anns), self.max_objs)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.opt.phase == "train" or self.opt.debug > 0:
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)

        if np.random.random() < 0.5:
            prob = 0.05
            rnd = np.random.rand(inp.shape[0], inp.shape[1], inp.shape[2])
            inp[rnd < prob] = np.random.rand(1)

        inp = (inp.astype(np.float32) / 255.)

        if self.opt.phase == 'train' and not self.opt.no_color_aug or self.opt.debug > 0:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio

        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._label_box_to_bbox(ann)
            cls_id = self.class_to_ind[ann[4]]
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius

                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})

        if self.opt.debug > 0 and self.opt.phase == 'test':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det}
            ret['meta'] = meta

        return ret
