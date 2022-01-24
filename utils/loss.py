import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, weight=None, reduction='mean', batch_average=False, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.batch_average = batch_average  # When 'reduction' is set to 'mean', 'batch_average' is redundant.
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'wb':
            return self.WeightedBalanceLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def WeightedBalanceLoss(self, logit, target, T=0.4):
        # n, c, h, w = logit.size()
        #
        # m = nn.LogSoftmax(dim=1)
        # logit_prob = m(logit)
        #
        # logit_label = torch.argmax(logit_prob, dim=1)
        # w1 = torch.zeros((n, h, w)).cuda()
        # w1[(target.long() - logit_label) == 1] = 1
        # w1 = w1.unsqueeze(1)
        #
        # w2 = torch.ones((n, h, w))*T
        # w2 = np.maximum(w2, torch.exp(logit_prob[:, 1, :, :]).cpu().detach())
        # w2 = w2.cuda().unsqueeze(1)
        #
        # target = target.unsqueeze(1)
        # loss = torch.mean(-w1*target*logit_prob[:, 1, :, :] - w2*(1-target)*logit_prob[:, 0, :, :])
        #
        # if self.batch_average:
        #     loss /= n
        #
        # return loss
        pass


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(4, 2, 7, 7).cuda()
    b = torch.rand(4, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print(loss.WeightedBalanceLoss(a, b).item())

    a = torch.rand(4, 2, 7, 7).cuda()
    b = torch.rand(4, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

    a = torch.rand(4, 2, 7, 7).cuda()
    b = torch.rand(4, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())


