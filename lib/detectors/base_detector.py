from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model


class BaseDetector(object):
	def __init__(self, opt):
		if opt.gpus[0] >= 0:
		  opt.device = torch.device('cuda')
		else:
		  opt.device = torch.device('cpu')

		print('Creating model...')
		
		self.model = create_model(model_name=opt.model_name,backbone=opt.backbone,opt=opt)		
		self.model = load_model(model=self.model, model_path=opt.load_model,resume=opt.resume)
		self.model = self.model.to(opt.device)	
		self.model.eval()
		self.num_classes = opt.num_classes
		self.opt = opt
		self.pause = True

	def pre_process(self, image):
		raise NotImplementedError

	def process(self, images, return_time=False):
		raise NotImplementedError

	def post_process(self):
		raise NotImplementedError

	def merge_outputs(self, detections):
		raise NotImplementedError

	def run(self, image_or_path_or_tensor):
		raise NotImplementedError


