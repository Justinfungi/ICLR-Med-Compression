#!/usr/bin/env python3
"""
TiTok MRI Fine-tuning Loss Functions
基于TiTok Stage 2的完整损失实现，适配MRI微调场景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import einops


class PerceptualLoss(nn.Module):
    """感知损失 - 使用预训练的特征提取器"""

    def __init__(self, net_type='vgg19', layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        self.net_type = net_type
        self.layers = layers

        # 使用torchvision的预训练模型
        if net_type == 'vgg19':
            from torchvision.models import vgg19
            self.model = vgg19(pretrained=True).features
        elif net_type == 'vgg16':
            from torchvision.models import vgg16
            self.model = vgg16(pretrained=True).features
        else:
            raise ValueError(f"Unsupported net_type: {net_type}")

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # 定义用于计算损失的层
        self.layer_weights = {
            'relu1_2': 1/32,
            'relu2_2': 1/16,
            'relu3_3': 1/8,
            'relu4_3': 1/4
        }

    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失

        Args:
            input_img: 输入图像 [B, C, H, W]
            target_img: 目标图像 [B, C, H, W]

        Returns:
            感知损失值
        """
        # 归一化到ImageNet标准
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_img.device)

        input_norm = (input_img - mean) / std
        target_norm = (target_img - mean) / std

        # 如果是单通道(灰度)图像，重复到3通道
        if input_norm.shape[1] == 1:
            input_norm = einops.repeat(input_norm, 'b 1 h w -> b 3 h w', c=3)
            target_norm = einops.repeat(target_norm, 'b 1 h w -> b 3 h w', c=3)

        features_input = self._extract_features(input_norm)
        features_target = self._extract_features(target_norm)

        loss = 0.0
        for layer_name in self.layers:
            if layer_name in features_input and layer_name in features_target:
                weight = self.layer_weights.get(layer_name, 1.0)
                loss += weight * F.mse_loss(features_input[layer_name], features_target[layer_name])

        return loss

    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取中间层特征"""
        features = {}
        layer_names = {
            1: 'relu1_1', 3: 'relu1_2',
            6: 'relu2_1', 8: 'relu2_2',
            11: 'relu3_1', 13: 'relu3_2', 15: 'relu3_3', 17: 'relu3_4',
            20: 'relu4_1', 22: 'relu4_2', 24: 'relu4_3', 26: 'relu4_4',
            29: 'relu5_1', 31: 'relu5_2', 33: 'relu5_3', 35: 'relu5_4'
        }

        for name, module in self.model.named_children():
            x = module(x)
            if int(name) in layer_names:
                features[layer_names[int(name)]] = x

        return features


class NLayerDiscriminator(nn.Module):
    """N层判别器 - 用于GAN训练"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator"""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(logits_real_mean: torch.Tensor, logits_fake_mean: torch.Tensor,
                      ema_logits_real_mean: torch.Tensor, ema_logits_fake_mean: torch.Tensor) -> torch.Tensor:
    """LeCam regularization loss"""
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class TiTokMRILoss(nn.Module):
    """
    TiTok MRI微调完整损失函数
    基于TiTok Stage 2，适配MRI图像特点
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # 损失权重配置
        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        self.discriminator_weight = config.get('discriminator_weight', 0.5)
        self.discriminator_factor = config.get('discriminator_factor', 1.0)
        self.discriminator_start = config.get('discriminator_start', 1000)
        self.lecam_regularization_weight = config.get('lecam_regularization_weight', 0.01)
        self.lecam_ema_decay = config.get('lecam_ema_decay', 0.999)

        # 损失类型
        self.reconstruction_loss_type = config.get('reconstruction_loss_type', 'l2')

        # 初始化组件
        self.perceptual_loss = PerceptualLoss(
            net_type=config.get('perceptual_net_type', 'vgg16'),
            layers=config.get('perceptual_layers', ['relu2_2', 'relu3_3'])
        )

        # GAN判别器 (可选)
        self.use_gan = config.get('use_gan', False)
        if self.use_gan:
            self.discriminator = NLayerDiscriminator(
                input_nc=config.get('discriminator_input_nc', 1),  # MRI通常是单通道
                ndf=config.get('discriminator_ndf', 32),
                n_layers=config.get('discriminator_layers', 3)
            )

            # LeCam EMA buffers
            if self.lecam_regularization_weight > 0.0:
                self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
                self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                global_step: int = 0,
                mode: str = "generator") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算完整损失

        Args:
            inputs: 原始图像 [B, C, H, W]
            reconstructions: 重建图像 [B, C, H, W]
            global_step: 当前训练步数
            mode: "generator" 或 "discriminator"

        Returns:
            total_loss: 总损失
            loss_dict: 各损失组件字典
        """

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, global_step)
        elif mode == "discriminator" and self.use_gan:
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _forward_generator(self, inputs: torch.Tensor, reconstructions: torch.Tensor, global_step: int):
        """生成器训练步"""

        # 1. 重建损失
        if self.reconstruction_loss_type == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss_type == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsupported reconstruction_loss_type: {self.reconstruction_loss_type}")

        reconstruction_loss *= self.reconstruction_weight

        # 2. 感知损失
        perceptual_loss = self.perceptual_loss(inputs, reconstructions)
        perceptual_loss *= self.perceptual_weight

        # 3. GAN损失 (如果启用)
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if global_step >= self.discriminator_start else 0.0

        if self.use_gan and discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # 冻结判别器梯度
            for param in self.discriminator.parameters():
                param.requires_grad = False

            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight = self.discriminator_weight

        # 4. 总损失
        total_loss = (
            reconstruction_loss
            + perceptual_loss
            + d_weight * discriminator_factor * generator_loss
        )

        # 5. 损失字典
        loss_dict = {
            'total_loss': total_loss.clone().detach(),
            'reconstruction_loss': reconstruction_loss.detach(),
            'perceptual_loss': perceptual_loss.detach(),
            'weighted_gan_loss': (d_weight * discriminator_factor * generator_loss).detach(),
            'discriminator_factor': torch.tensor(discriminator_factor),
            'd_weight': torch.tensor(d_weight),
            'gan_loss': generator_loss.detach(),
        }

        return total_loss, loss_dict

    def _forward_discriminator(self, inputs: torch.Tensor, reconstructions: torch.Tensor, global_step: int):
        """判别器训练步"""

        discriminator_factor = self.discriminator_factor if global_step >= self.discriminator_start else 0

        # 启用判别器梯度
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        fake_images = reconstructions.detach()

        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(fake_images)

        # Hinge损失
        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real, logits_fake)

        # LeCam正则化
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            # 更新EMA
            self.ema_real_logits_mean = (
                self.ema_real_logits_mean * self.lecam_ema_decay +
                torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            )
            self.ema_fake_logits_mean = (
                self.ema_fake_logits_mean * self.lecam_ema_decay +
                torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
            )

        discriminator_loss += lecam_loss

        loss_dict = {
            'discriminator_loss': discriminator_loss.detach(),
            'logits_real': logits_real.detach().mean(),
            'logits_fake': logits_fake.detach().mean(),
            'lecam_loss': lecam_loss.detach(),
        }

        return discriminator_loss, loss_dict

    def should_discriminator_be_trained(self, global_step: int) -> bool:
        """检查是否应该训练判别器"""
        return global_step >= self.discriminator_start and self.use_gan


class SimpleMRILoss(nn.Module):
    """
    简化的MRI损失函数 - 只包含重建和感知损失
    适合资源有限的微调场景
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        self.reconstruction_loss_type = config.get('reconstruction_loss_type', 'l2')

        # 简化的感知损失 (只用一层特征)
        self.perceptual_loss = PerceptualLoss(
            net_type=config.get('perceptual_net_type', 'vgg16'),
            layers=['relu3_3']  # 只用一层减少计算量
        )

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""

        # 重建损失
        if self.reconstruction_loss_type == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss_type == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsupported reconstruction_loss_type: {self.reconstruction_loss_type}")

        reconstruction_loss *= self.reconstruction_weight

        # 感知损失
        perceptual_loss = self.perceptual_loss(inputs, reconstructions)
        perceptual_loss *= self.perceptual_weight

        # 总损失
        total_loss = reconstruction_loss + perceptual_loss

        loss_dict = {
            'total_loss': total_loss.clone().detach(),
            'reconstruction_loss': reconstruction_loss.detach(),
            'perceptual_loss': perceptual_loss.detach(),
        }

        return total_loss, loss_dict
