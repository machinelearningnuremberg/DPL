import torch
import torch.nn as nn


class NN(nn.Module):

    def __init__(
        self,
        nr_initial_features=10,
        nr_units=200,
        nr_layers=3,
        dropout_fraction=0.2,
        nr_classes=1,
    ):
        """

        Args:
            nr_initial_features: int
                The number of features per example.
            nr_units: int
                The number of units for every layer.
            nr_layers: int
                The number of layers for the neural network.
            dropout_fraction: float
                The dropout fraction to be used through training.
            nr_classes: int
                The number of classes in the dataset.
        """
        super(NN, self).__init__()
        self.nr_layers = nr_layers
        self.fc1 = nn.Linear(nr_initial_features, nr_units)
        self.bn1 = nn.BatchNorm1d(nr_units)
        for i in range(2, nr_layers + 1):
            setattr(self, f'fc{i}', nn.Linear(nr_units, nr_units))
            setattr(self, f'bn{i}', nn.BatchNorm1d(nr_units))
        setattr(self, f'fc{nr_layers + 1}', nn.Linear(nr_units, nr_classes))

        self.dropout = nn.Dropout(p=dropout_fraction)
        self.last_act_func = torch.nn.LeakyReLU()


    def forward(self, x):

        x = x.view(-1, self.num_flat_features(x))

        x = self.last_act_func(self.bn1(self.fc1(x)))
        for i in range(2, self.nr_layers + 1):
            x = self.dropout(x)
            temp_layer = getattr(self, f'fc{i}')
            x = self.last_act_func(getattr(self, f'bn{i}')(temp_layer(x)))

        x = self.dropout(x)
        x = getattr(self, f'fc{self.nr_layers + 1}')(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
