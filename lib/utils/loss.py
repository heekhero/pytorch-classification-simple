import torch

def softmax_cross_entropy_loss(input, target):
    assert input.dim() == 2
    assert input.size()
    target = target.long()

    e_input = torch.exp(input)
    inds = torch.arange(len(target)).long()
    loss = -torch.log(e_input[inds, target] / torch.sum(e_input, dim=1))
    loss = loss.mean()

    return loss

