import torch
import torch.nn as nn


class ConditionedPowerLawV1(nn.Module):

    def __init__(
        self,
        nr_initial_features=10,
        nr_units=200,
        nr_layers=3,
        use_learning_curve: bool = True,
        kernel_size: int = 3,
        nr_filters: int = 4,
        nr_cnn_layers: int = 2,
    ):
        """
        Args:
            nr_initial_features: int
                The number of features per example.
            nr_units: int
                The number of units for every layer.
            nr_layers: int
                The number of layers for the neural network.
            use_learning_curve: bool
                If the learning curve should be use in the network.
            kernel_size: int
                The size of the kernel that is applied in the cnn layer.
            nr_filters: int
                The number of filters that are used in the cnn layers.
            nr_cnn_layers: int
                The number of cnn layers to be used.
        """
        super(ConditionedPowerLaw, self).__init__()

        self.use_learning_curve = use_learning_curve
        self.kernel_size = kernel_size
        self.nr_filters = nr_filters
        self.nr_cnn_layers = nr_cnn_layers

        self.act_func = torch.nn.LeakyReLU()
        self.relu_func = torch.nn.ReLU()
        self.last_act_func = torch.nn.GLU()
        self.tan_func = torch.nn.Tanh()
        self.batch_norm = torch.nn.BatchNorm1d

        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = nr_initial_features
        if self.use_learning_curve:
            nr_initial_features = nr_initial_features + nr_filters

        layers.append(nn.Linear(nr_initial_features, nr_units))
        layers.append(self.act_func)

        for i in range(2, nr_layers + 1):
            layers.append(nn.Linear(nr_units, nr_units))
            layers.append(self.act_func)

        last_layer = nn.Linear(nr_units, 4)

        layers.append(last_layer)
        with torch.no_grad():
            last_layer.bias.data = torch.Tensor([0.1, 1, 0.2, 0.2])

        self.layers = torch.nn.Sequential(*layers)

        cnn_part = []
        if use_learning_curve:
            cnn_part.append(
                nn.Conv1d(
                    in_channels=2,
                    kernel_size=(self.kernel_size,),
                    out_channels=self.nr_filters,
                ),
            )
            for i in range(1, self.nr_cnn_layers):
                cnn_part.append(self.act_func)
                cnn_part.append(
                    nn.Conv1d(
                        in_channels=self.nr_filters,
                        kernel_size=(self.kernel_size,),
                        out_channels=self.nr_filters,
                    ),
                ),
            cnn_part.append(nn.AdaptiveAvgPool1d(1))

        self.cnn = nn.Sequential(*cnn_part)

    def forward(
        self,
        x: torch.Tensor,
        predict_budgets: torch.Tensor,
        evaluated_budgets: torch.Tensor,
        learning_curves: torch.Tensor,
    ):
        """
        Args:
            x: torch.Tensor
                The examples.
            predict_budgets: torch.Tensor
                The budgets for which the performance will be predicted for the
                hyperparameter configurations.
            evaluated_budgets: torch.Tensor
                The budgets for which the hyperparameter configurations have been
                evaluated so far.
            learning_curves: torch.Tensor
                The learning curves for the hyperparameter configurations.
        """
        #x = torch.cat((x, torch.unsqueeze(evaluated_budgets, 1)), dim=1)
        if self.use_learning_curve:
            lc_features = self.cnn(learning_curves)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)
            x = torch.cat((x, lc_features), dim=1)

        x = self.layers(x)
        a = x[:, 0]
        b = x[:, 1]
        c = x[:, 2]
        d = x[:, 3]

        first_part = self.relu_func(
            torch.add(
                predict_budgets,
                d,
            ),
        )

        output = torch.add(
            a,
            torch.mul(
                b,
                torch.pow(
                    first_part,
                    -c,
                ),
            ),
        )

        return output
