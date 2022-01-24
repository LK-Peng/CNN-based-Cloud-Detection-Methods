import numpy as np

# 忽略除数为0的warning, 对于除数为0的情况已有相应处理
np.seterr(divide='ignore', invalid='ignore')

from utils.f_boundary import eval_mask_boundary


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Precision(self):
        assert self.num_class == 2
        pr = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1])
        return 1.0 if np.isnan(pr) else pr

    def Recall(self):
        assert self.num_class == 2
        re = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0])
        return 1.0 if np.isnan(re) else re

    def F_score(self):
        assert self.num_class == 2
        pr, re = self.Precision(), self.Recall()
        if pr + re == 0:
            return 0.0
        else:
            return 2.0 * pr * re / (pr + re)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        return MIoU[1]

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class BoundaryEvaluator(object):
    def __init__(self, num_class, p=None, num_proc=10, bound_th=0.008):
        self.num_class = num_class
        self.p = p
        self.num_proc = num_proc
        self.bound_th = bound_th
        self.confusion_matrix_pc = np.zeros((self.num_class, 4))

    def Precision_boundary(self):
        pr = self.confusion_matrix_pc[:, 0] / self.confusion_matrix_pc[:, 1]
        pr[np.isnan(pr)] = 1.0
        return pr

    def Recall_boundary(self):
        re = self.confusion_matrix_pc[:, 2] / self.confusion_matrix_pc[:, 3]
        re[np.isnan(re)] = 1.0
        return re

    def F_score_boundary(self):
        pr, re = self.Precision_boundary(), self.Recall_boundary()
        f_score = 2 * pr * re / (pr + re)
        f_score[np.isnan(f_score)] = 0.0
        return f_score

    def _generate_matrix(self, gt_image, pre_image):
        if len(gt_image.shape) == 2:
            pre_image, gt_image = np.expand_dims(pre_image, axis=0), np.expand_dims(gt_image, axis=0)
        return eval_mask_boundary(pre_image, gt_image, self.num_class, self.p, self.num_proc, self.bound_th)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix_pc += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix_pc = np.zeros((self.num_class, 4))
