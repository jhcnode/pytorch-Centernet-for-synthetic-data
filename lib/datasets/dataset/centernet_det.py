from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import torch.utils.data as data

class CenternetDetDataset(data.Dataset):
	num_classes = -1
	default_resolution = [512, 512]
	mean = np.array([0.40789654, 0.44719302, 0.47026115],
				   dtype=np.float32).reshape(1, 1, 3)
	std  = np.array([0.28863828, 0.27408164, 0.27809835],
				   dtype=np.float32).reshape(1, 1, 3)

	def __init__(self,opt,gt_labels,class_name):
		super(CenternetDetDataset, self).__init__()
		self.data_dir=opt.data_dir
		self.max_objs = 128
		self._data_rng = np.random.RandomState(123)
		self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
								 dtype=np.float32)
		self._eig_vec = np.array([
			[-0.58752847, -0.69563484, 0.41340352],
			[-0.5832747, 0.00994535, -0.81221408],
			[-0.56089297, 0.71832671, 0.41158938]
		], dtype=np.float32)
		opt.class_name=class_name
		self.opt=opt
		self.gt_labels = gt_labels
		self.class_name = opt.class_name
		self.num_classes=len(self.class_name)
		self.class_to_ind = dict(zip(self.class_name, range(self.num_classes)))	
		self.num_samples=len(self.gt_labels)
		
	def __len__(self):
		return self.num_samples

