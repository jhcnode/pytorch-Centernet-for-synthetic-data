from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .centernetdet_trainer import CenternetDetTrainer


train_factory = {
	'ctdet': CenternetDetTrainer,
	'ctdetp': CenternetDetTrainer
}
