import torch
import torch.nn as nn
from collections import OrderedDict

class ACoL(nn.Module):
    def __init__(self, base_model, cls_recipe, nb_classes, deltas, device):
        super(ACoL, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.cls_recipe = list(cls_recipe)

        cls_layers = OrderedDict()
        for idx, cls in enumerate(cls_recipe):
            cls_layers[str(idx) + cls] = self.generate_classifier(nb_classes)

        self.classifiers = nn.Sequential(cls_layers)
        self.deltas = [float(d) for d in deltas]

    def generate_classifier(self, nb_classes):
        cls = nn.Sequential(
            nn.Conv2d(2048, nb_classes, kernel_size=(1, 1), stride=1),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        )
        return cls

    def generate_cam(self, idx, _s, labels):
        W_conv = self.classifiers[idx][0].weight.detach()
        W_conv_c = W_conv[labels]

        A_conv_c = (_s * W_conv_c).sum(1)

        # normalize
        _min = A_conv_c.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
        _max = A_conv_c.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        A_conv_c_normalized = (A_conv_c - _min) / (_max - _min)

        return A_conv_c_normalized

    def forward(self, inputs, labels):
        x = self.backbone(inputs)

        cls_output = []
        cams = []
        for idx, cls in enumerate(self.cls_recipe):
            output = self.classifiers[idx](x).squeeze(-1).squeeze(-1)

            cls_output.append(output)
            cam = self.generate_cam(idx, x, labels)
            cams.append(cam)

            if idx < len(self.cls_recipe) - 1:

                mask = (cam > self.deltas[idx]).unsqueeze(1).byte()
                x = x.masked_fill(mask, value=0)

            else:
                v_cls_output = torch.stack(cls_output)
                v_cams = torch.stack(cams)
                return v_cls_output, v_cams