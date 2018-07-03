import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as vmodels
import torchvision.datasets as vdatasets

from acol import ACoL
from utils import produce_intermediate_result


import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(42)

import time, copy, os


def restCrossEtropyLoss(X, y, device):
    _base = torch.ones(X.size()).to(device)
    _one_hot = _base.scatter(1, y.view(-1, 1), 0)
    C = _one_hot[0].sum()

    denom = torch.exp(X).sum(1)
    numer = torch.exp((X * _one_hot).sum(1) / C)

    loss = -torch.log(numer / denom).mean()

    return loss

def run(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ready dataset
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(197),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((197, 197)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_dir = "../input/"
    image_datasets = {x: vdatasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'validation']}

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True),
                   'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle=False)}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    nb_classes = len(image_datasets['train'].classes)


    # build model
    if args.model == 'resnet50':
        model_ft = vmodels.resnet50(pretrained=True)
    elif args.model == 'resnet101':
        model_ft = vmodels.resnet101(pretrained=True)
    elif args.model == 'resnet152':
        model_ft = vmodels.resnet152(pretrained=True)
    else:
        raise ValueError('Unable to load pretrained model.')

    deltas = [float(d) for d in args.delta_list.split(",")]
    acol = ACoL(model_ft, args.cls_recipe, nb_classes, deltas, device)
    acol.to(device)


    # loss_function and optimizer
    loss_function = nn.CrossEntropyLoss()
    opt = optim.SGD(acol.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)


    # train and evaluate
    def train_model(model, loss_function, optimizer, num_epochs=30):
        since = time.time()

        best_acc = 0.0

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()

                    print(model.training)
                    print(model.backbone.training)
                    for cls in model.classifiers:
                        print(cls.training)

                else:
                    model.eval()

                    print(model.training)
                    print(model.backbone.training)
                    for cls in model.classifiers:
                        print(cls.training)

                running_loss = 0.0
                running_corrects = 0

                counter = 0
                tmp = {}

                for inputs, labels in dataloaders[phase]:

                    counter += 1

                    current_label_set = labels.numpy().tolist()

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):

                        # forward
                        outputs, cams = model(inputs, labels)
                        label_prob = F.softmax(outputs, dim=2)

                        # pred = first classifier(p) output
                        _, preds = torch.max(outputs[0], 1)


                        # calculate loss
                        total_loss = torch.tensor([0.0]).to(device)
                        for idx, cls in enumerate(model.cls_recipe):

                            if cls == 'p':
                                loss = loss_function(outputs[idx], labels)
                            elif cls == 'n':
                                loss = restCrossEtropyLoss(outputs[idx], labels, device)
                            else:
                                raise ValueError("invalid cls recipe")
                            total_loss += loss

                        # backward and update optimizer
                        if phase == 'train':

                            if counter % 10 == 0:
                                print("| e-{:03d} | i-{:03d} | loss: {:.4f} |"
                                      .format(epoch, counter, loss.item()))
                            total_loss.backward()
                            optimizer.step()

                        # visualize cams
                        if phase == 'validation':

                            if len(set(current_label_set)) == 1:
                                if current_label_set[0] not in tmp.keys():

                                    print("val ", counter)
                                    print("saving image")
                                    produce_intermediate_result(inputs, labels, cams, args.cls_recipe,
                                                                label_prob,
                                                                epoch, counter, device)
                                    tmp[current_label_set[0]] = 'added'

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # metric for epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc))
                print('{{"metric": "{}_loss", "value": {}}}'.format(phase, epoch_loss))
                print('{{"metric": "{}_acc", "value": {}}}'.format(phase, epoch_acc))

                # save the model when val acc is updated
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc

                    if epoch > 3:
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(best_model_wts,
                                   "/output/best_model_e{:02d}_val_acc{:.2f}.pth.tar".format(epoch, epoch_acc))


            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(best_acc))

    train_model(acol, loss_function, opt, num_epochs=args.epochs)


def main(argv):

    parser = argparse.ArgumentParser(description="PyTorch ACoL Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--epochs", type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10')
    parser.add_argument("--model", type=str, default='resnet50',
                        help="choose base model - resnet50 or resnet101 or resnet152")
    parser.add_argument("--cls-recipe", type=str, default='pp',
                        help="classifier model seqeuence - p for positive, n for negative (default: pp)")
    parser.add_argument("--delta-list", type=str, default='0.9',
                        help="list of delta thresholds for masking (default: 0.9)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])