import numpy as np
import os
import torch.utils.data as data# 축약금지



class CenternetDetPatchDataset(data.Dataset):
	default_resolution = [512, 512]
	mean = np.array([0, 0, 0],
				   dtype=np.float32).reshape(1, 1, 3)
	std  = np.array([1, 1, 1],
				   dtype=np.float32).reshape(1, 1, 3)

	def __init__(self,opt,patch_labels,back_labels,class_name,negative_labels=None,dataset_capacity=8192):
		super(CenternetDetPatchDataset, self).__init__()
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
		self.patch_labels = patch_labels
		self.back_labels = back_labels
		self.negative_labels=negative_labels
		self.class_name = opt.class_name
		self.num_classes=len(self.class_name)
		self.class_to_ind = dict(zip(self.class_name, range(self.num_classes)))
		self.dataset_capacity=dataset_capacity
	def __len__(self):
		return self.dataset_capacity


