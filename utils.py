import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision import transforms as transforms

vis = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((197, 197)),
        transforms.ToTensor()
    ])

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)


def produce_intermediate_result(inputs, labels, cams, cls_seq, label_prob, epoch, counter, device):
    base = torch.zeros(label_prob[0].size()).to(device)
    tmp = base.scatter(1, labels.view(-1, 1), 1)
    one_hot = torch.stack([tmp] * label_prob.size()[0])
    target_prob = (label_prob * one_hot).sum(2)
    cls_list = list(cls_seq)

    resized_cams = []
    print(cams.size())
    for idx, cls in enumerate(cls_list):
        r_cam = torch.stack([vis(m.unsqueeze(0).cpu()) for m in cams[idx]])
        resized_cams.append(r_cam)

    # input, nb_cls, fused
    ncol = 1 + len(cls_list) + 1
    nrow = 4
    fig, ax = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))

    for i in range(nrow):
        org_img = inv_normalize(inputs[i]).cpu().numpy().transpose(1, 2, 0)
        ax[i, 0].imshow(org_img)
        ax[i, 0].set_title("label: {}".format(labels[i].item()))
        fused = np.zeros(org_img.shape[:2])
        for idx, cls in enumerate(cls_list):
            current_cam = resized_cams[idx][i][0].numpy()
            ax[i, idx + 1].imshow(current_cam, cmap='jet')
            ax[i, idx + 1].set_title("{}: {:.4f}".format(cls, target_prob[idx][i]))

            if cls == 'p':
                fused = np.maximum(fused, current_cam)

        ax[i, 1 + len(cls_list)].imshow(org_img)
        ax[i, 1 + len(cls_list)].imshow(fused, cmap='jet', alpha=0.4)
        ax[i, 1 + len(cls_list)].set_title("fused")

    for a in ax:
        for b in a:
            b.axis("off")

    fig.suptitle("epoch_{:02d}".format(epoch))

    fname = "/output/fused_e{:02d}_i{:03d}.png".format(epoch, counter)
    fig.savefig(fname, bbox_inches='tight')