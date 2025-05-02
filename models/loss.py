# python import
from typing import Union
# package import
import torch
import torch.nn as nn
import torchvision.models as models
# local import


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)

        self.feature_extractor = vgg.features

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, generated_image, target_image):
        b = generated_image.shape[0]
        if generated_image.shape[1] == 1:
            generated_image_3ch = generated_image.repeat(1, 3, 1, 1)
            target_image_3ch = target_image.repeat(1, 3, 1, 1)
        else:
            generated_image_3ch = generated_image
            target_image_3ch = target_image

        perceptual_loss = 0.0

        for i, layer in enumerate(self.feature_extractor):
                
            generated_features = layer(generated_image_3ch)
            target_features = layer(target_image_3ch)

            # if i in [2, 7, 12, 21, 30]:  # VGG16 的特征层索引
            if i in [2, 12, 30]:  # VGG16 的特征层索引
                perceptual_loss += self.mse_loss(generated_features, target_features)

                generated_image_3ch = generated_features.clone()
                target_image_3ch = target_features.clone()
            else:
                generated_image_3ch = generated_features
                target_image_3ch = target_features
        return perceptual_loss / (3 * b)

        # # 提取特征
        # generated_features = self.feature_extractor(generated_image_3ch)
        # target_features = self.feature_extractor(target_image_3ch)

        # # 计算特征的均方误差
        # perceptual_loss = self.mse_loss(generated_features, target_features)

        # return perceptual_loss


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
