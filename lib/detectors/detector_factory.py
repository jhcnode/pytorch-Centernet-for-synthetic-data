from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetDetector
from .centernetdet_detector import CenternetDetDetector

detector_factory = {
	'ctdet': CtdetDetector,
	'ctdetp': CenternetDetDetector
}
