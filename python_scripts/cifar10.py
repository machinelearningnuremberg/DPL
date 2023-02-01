from __future__ import print_function
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.preact_resnet import PreActResNet18


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR10 benchmarking')
    parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--nr_epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--output_path', type=str, default='./benchmarks/cifar',
                        help='Path where the results will be stored.')
    parser.add_argument('--index', type=int, default=1,
                        help='The index of the dropout rate to be used.')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    file_path = os.path.join(
        args.output_path,
        f'hp_config{args.index}.log',
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    drop_rate_possible_values = np.arange(0.05, 0.85, 0.05)
    drop_rate = drop_rate_possible_values[args.index]
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=1,
    )

    test_set = datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1,
    )

    criterion = nn.CrossEntropyLoss()
    model = PreActResNet18(drop_rate).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nr_epochs)
    test_epoch_performances = []
    for epoch in range(args.nr_epochs):

        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 99:.3f}')
                running_loss = 0.0

        model.eval()
        # total number of examples evaluated
        total = 0
        # number of correctly classified examples
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        fraction_correct_examples = correct / total
        test_epoch_performances.append(fraction_correct_examples)
        with open(file_path, 'w') as fp:
            json.dump({drop_rate: test_epoch_performances}, fp)

        scheduler.step(epoch + 1)

    if args.save_model:
        torch.save(
            model.state_dict(),
            os.path.join(args.output_path, f'{drop_rate}_cifar10_cnn.pt'),
        )

if __name__ == '__main__':
    main()
