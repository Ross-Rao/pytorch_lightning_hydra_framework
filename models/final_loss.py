# python import
from typing import Union
import torch


def auto_scaled_loss(loss_lt: Union[list[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(loss_lt, torch.Tensor):
        loss_lt = [loss_lt]
    if len(loss_lt) == 1:
        return loss_lt[0]
    elif len(loss_lt) == 0:
        raise ValueError("No loss to scale")

    final_loss = loss_lt[0]
    for i in range(1, len(loss_lt)):
        if loss_lt[i] > 0:
            final_loss += loss_lt[i] / (loss_lt[i] / loss_lt[0]).detach()
    return final_loss
