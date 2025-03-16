# python import
# package import
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
# local import
from models.encoder import ResNetEncoder, TransformerEncoder
from models.decoder import SimpleConvDecoder
from models.final_loss import auto_scaled_loss

__all__ = ['EncoderDecoderModel', 'ModelLoss']


class MemoryBank(torch.autograd.Function):
    """
    update the memory bank with the new data
    """

    @staticmethod
    def forward(ctx, x, index, memory, params):
        # only train sample's index should be stored to update memory
        # if index is None, it should be external test samples
        # if index is not None and index.max() < memory.size(0), it should be val samples
        if index is not None and index.max() < memory.size(0):
            # use clone to avoid modifying input(x and index) in multiprocess environment
            ctx.save_for_backward(x.clone(), index.clone(), memory, params)
        t = params[0]  # temperature scaler

        output = x @ memory.t() / t
        return output  # only x is needed for backward

    @staticmethod
    def backward(ctx, *grad_output):
        x, index, memory, params = ctx.saved_tensors
        t, momentum = params[0], params[1]

        grad_x = grad_output[0]
        grad_input = grad_x @ memory / t

        # update memory
        weight = memory[index] * momentum + x * (1. - momentum)
        memory[index] = F.normalize(weight, p=2, dim=1)  # normalize memory

        return grad_input, None, None, None


class MemoryCluster(nn.Module):
    """
    use memory cluster to calculate the instance loss and anchor loss
    """

    def __init__(self, n_samples, npc_dim, neighbors=1, temperature=0.07, momentum=0.5, const=0.):
        super().__init__()
        self.samples_num = n_samples
        self.npc_dim = npc_dim
        self.neighbors_num = neighbors
        self.const = const

        # the memory bank should only store train samples
        # indices are reset by order: train, val, test
        # so if input index > n_samples, it should be val or internal test samples
        # if input index is None, it should be external test samples
        # it works by:
        # 1. `split_dataset_folds` by param `reset_split_index` in modules/monai_data_module.py
        # 2. `IndexTransformd` in utils/custom_transforms.py
        self.register_buffer('memory', torch.rand(n_samples, npc_dim))
        std = 1. / torch.sqrt(torch.tensor(npc_dim / 3.))
        self.memory = self.memory * 2 * std - std
        self.register_buffer('params', torch.tensor([temperature, momentum]))

        # anchor samples' flag >= 0, instance samples' flag < 0
        self.register_buffer('flag', -torch.arange(n_samples).long() - 1)

        # neighbors
        self.register_buffer('neighbors', torch.LongTensor(n_samples))

        self.register_buffer('entropy', torch.zeros(n_samples))  # use local variable might error for device

    def update_anchor(self, search_rate, mini_batch_size=256):
        with torch.no_grad():
            # calculate entropy, use for loop to save gpu memory
            for start in range(0, self.samples_num, mini_batch_size):
                end = min(start + mini_batch_size, self.samples_num)
                sim = MemoryBank.apply(self.memory[start:end], None, self.memory, self.params)
                pred = F.softmax(sim, dim=1)  # mini_batch_size * n_samples
                self.entropy[start:end] = -torch.sum(pred * torch.log(pred + self.const), dim=1)

            # update anchor
            anchor_nums = int(self.samples_num * search_rate)  # search range expand with epoch
            sort = torch.argsort(self.entropy)
            anchor = sort[:anchor_nums]  # Lower entropy indicates a higher similarity
            instance = sort[anchor_nums:]
            self.flag[anchor] = 1
            self.flag[instance] = -1

            # using anchor feature to find neighbors, for loop to save gpu memory
            for start in range(0, anchor_nums, mini_batch_size):
                end = min(start + mini_batch_size, anchor_nums)
                anchor_feature = self.memory[anchor[start:end]]
                sim = MemoryBank.apply(anchor_feature, None, self.memory, self.params)
                pred = F.softmax(sim, dim=1)
                # remove self-similarity
                pred[torch.arange(end - start), anchor[start:end]] = -1
                # find the nearest neighbor (should be written by torch.max)
                sort = torch.argmax(pred, dim=1)
                self.neighbors[anchor[start:end]] = sort

    def forward(self, zp, index, local_neighbor_indices=None):
        if index is not None and index.max() < self.samples_num:  # train sample to anchor and instance loss
            # instance loss and anchor loss
            flags = self.flag[index]
            instance_indices = index[flags < 0]
            anchor_indices = index[flags >= 0]

            # non-parametric classification using memory bank
            zn = F.normalize(zp, p=2, dim=1)
            # For each image get similarity with neighbour
            sim = MemoryBank.apply(zn, index, self.memory, self.params)
            pred = F.softmax(sim, dim=1)

            batch_size = zp.size(0)

            if len(instance_indices) == 0:
                instance_loss = torch.tensor(0, dtype=torch.float32)
            else:
                pred_instance = pred[flags < 0, instance_indices]
                if local_neighbor_indices is not None:
                    # calculate local neighbor similarity
                    local_instance_neighbor_indices = local_neighbor_indices[flags < 0]
                    assert torch.all((instance_indices - local_instance_neighbor_indices) != 0)
                    pred_local_neighbor = pred[flags < 0, local_instance_neighbor_indices]

                    instance_loss = -torch.log(pred_instance + pred_local_neighbor + self.const).sum() / batch_size * 2
                else:
                    instance_loss = -torch.log(pred_instance + self.const).sum() / batch_size * 2

            if len(anchor_indices) == 0:
                anchor_loss = torch.tensor(0, dtype=torch.float32)
            else:
                # each anchor sample has neighbors
                anchor_neighbor_indices = self.neighbors[anchor_indices]  # global index
                # for each anchor sample, get similarity with neighbors
                # flags >= 0 is local index for anchor
                pred_neighbor = pred[flags >= 0, anchor_neighbor_indices]
                pred_anchor = pred[flags >= 0, anchor_indices]
                if local_neighbor_indices is not None:
                    # calculate local neighbor similarity
                    local_anchor_neighbor_indices = local_neighbor_indices[flags >= 0]
                    assert torch.all((anchor_indices - local_anchor_neighbor_indices) != 0)
                    pred_local_neighbor = pred[flags >= 0, local_anchor_neighbor_indices]

                    # remove local neighbor
                    pred_neighbor[anchor_neighbor_indices == local_anchor_neighbor_indices] = 0

                    anchor_loss = -torch.log(pred_anchor + pred_neighbor + pred_local_neighbor +
                                             self.const).sum() / batch_size * 2
                else:
                    anchor_loss = -torch.log(pred_anchor + pred_neighbor + self.const).sum() / batch_size * 2

            return instance_loss, anchor_loss

        else:  # validation and test sample don't need to calculate loss, so only return similar training sample's index
            assert not self.training, 'MemoryCluster should be in eval mode'
            zn = F.normalize(zp, p=2, dim=1)
            sim = MemoryBank.apply(zn, None, self.memory, self.params)
            pred = F.softmax(sim, dim=1)

            # similar training sample's index and similar training sample's flag
            indices = torch.argmax(pred, dim=1)  # training sample in memory
            flags = self.flag[indices]  # anchor or instance of the query sample
            return flags


class EncoderDecoderModel(nn.Module):
    def __init__(self,
                 stage,
                 n_classes=3,
                 encoder_in_channels=2,
                 decoder_out_channels=1,
                 decoder_layers=5,
                 pretrained=True,
                 encoder='resnet18',
                 hidden_dim=512,
                 npc_dim=128,
                 activation='ReLU'):
        super().__init__()
        self.stage = stage
        self.encoder_in_channels = encoder_in_channels

        self.encoder = ResNetEncoder(backbone=encoder, in_channels=encoder_in_channels,
                                     pretrained=pretrained, freeze_all=False)
        # 5 layers will turn 2^5 = 32 larger
        self.decoder = SimpleConvDecoder(in_channels=hidden_dim, out_channels=decoder_out_channels,
                                         layers=decoder_layers, activation=activation)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(hidden_dim, npc_dim)
        self.act = getattr(nn, activation)()
        self.transformer = TransformerEncoder(embedding_dim=npc_dim * 2, sequence_length=3, num_layers=decoder_layers,
                                              num_heads=8, positional_embedding='learnable')
        self.aff = nn.Linear(npc_dim, npc_dim)
        self.fc2 = nn.Linear(npc_dim * 2, n_classes)

    def forward(self, x, index, local_neighbor_indices):
        if x.size(1) != self.encoder_in_channels:
            b, c, h, w = x.size()
            x = x.reshape(-1, self.encoder_in_channels, h, w)
            if index.shape[0] != x.shape[0]:
                flatten_index = (c * index.unsqueeze(1) + torch.arange(c).to(index.device)).reshape(-1)
            else:
                flatten_index = index
            if local_neighbor_indices is None:
                flatten_local_neighbor_indices = (c * index.unsqueeze(1) +
                                                  torch.roll(torch.arange(c).to(index.device), 1)).reshape(-1)
            elif local_neighbor_indices.shape[0] != x.shape[0]:
                flatten_local_neighbor_indices = (c * local_neighbor_indices.unsqueeze(1) +
                                                  torch.arange(c).to(index.device)).reshape(-1)
            else:
                flatten_local_neighbor_indices = local_neighbor_indices
            cls, x_hat, zp, _, _ = self._forward(x, flatten_index, flatten_local_neighbor_indices)
            cls = cls.reshape(b, c, -1)
            x_hat = x_hat.reshape(b, c, h, w)
            zp = zp.reshape(b, c, -1)
            flatten_index = flatten_index.reshape(b, c)
            flatten_local_neighbor_indices = flatten_local_neighbor_indices.reshape(b, c)
            return cls, x_hat, zp, flatten_index, flatten_local_neighbor_indices
        else:
            return self._forward(x, index, local_neighbor_indices)

    def _forward(self, x, index, local_neighbor_indices):
        x = self.encoder(x).view((x.size(0), -1, 2, 2))
        x_hat = self.decoder(x)
        zb = self.pool(x).squeeze()  # reshape to hidden_dim
        zp = self.act(self.fc1(zb))
        if self.stage == 1:
            cls = torch.zeros_like(index)
        elif self.stage == 2:
            seq = torch.cat([zp, self.aff(zp)], dim=1)
            seq = seq.reshape(-1, 3, seq.shape[-1])
            cls = self.fc2(self.transformer(seq))
        else:
            raise ValueError('stage should be 1 or 2')
        return cls, x_hat, zp, index, local_neighbor_indices


class ModelLoss(nn.Module):
    def __init__(self,
                 stage,
                 n_samples,
                 npc_dim=128,
                 neighbors=1,
                 temperature=0.07,
                 momentum=0.5,
                 const=1e-8):
        super().__init__()
        self.stage = stage
        self.n_samples = n_samples
        self.recon_loss = nn.L1Loss(reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss()
        self.mc = MemoryCluster(n_samples, npc_dim, neighbors, temperature, momentum, const)

    def forward(self, pred_cls, x_hat, zp, index, local_neighbor_indices, cls):
        if len(index.shape) == 1:
            return self._forward(pred_cls, x_hat, zp, index, local_neighbor_indices, cls)
        else:
            b, c, _ = zp.shape
            pred_cls = pred_cls.reshape(-1, pred_cls.shape[-1])
            zp = zp.reshape(-1, zp.shape[-1])
            index = index.reshape(-1)
            local_neighbor_indices = local_neighbor_indices.reshape(-1)
            cls = cls.unsqueeze(1).repeat(1, c).reshape(-1)
            return self._forward(pred_cls, x_hat, zp, index, local_neighbor_indices, cls)

    def _forward(self, pred_cls, x_hat, zp, index, local_neighbor_indices, cls):
        if self.stage == 1:
            if index is not None and index.max() < self.n_samples:  # training loss
                instance_loss, anchor_loss = self.mc(zp, index, local_neighbor_indices)
                # recon_loss = self.recon_loss(x_mid, x_hat)
                loss_dt = {
                    # 'cls_loss': cls_loss,
                    # 'recon_loss': recon_loss,
                    'instance_loss': instance_loss,
                    'anchor_loss': anchor_loss
                }
                loss = auto_scaled_loss([instance_loss, anchor_loss])
                return loss, loss_dt
            else:
                # recon_loss = self.recon_loss(x_mid, x_hat)
                pred_flags = self.mc(zp, index)
                ratio = (pred_flags >= 0).float().mean()
                loss_dt = {
                    # 'recon_loss': recon_loss,
                    'anchor_ratio': ratio,
                }
                return 1 - ratio, loss_dt
        elif self.stage == 2:
            cls_loss = self.cls_loss(pred_cls, cls)  # only calculate classification loss
            loss_dt = {
                'cls_loss': cls_loss,
            }
            return cls_loss, loss_dt
        else:
            raise ValueError('stage should be 1 or 2')
