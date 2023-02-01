import torch


class PowerLaw(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.alpha = torch.nn.Parameter(torch.rand(()))
        self.beta = torch.nn.Parameter(torch.rand(()))
        self.gamma = torch.nn.Parameter(torch.rand(()))

        self.act_func = torch.nn.LeakyReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        output = torch.add(
            self.act_func(self.alpha),
            torch.mul(
                self.act_func(self.beta),
                torch.pow(
                    x,
                    torch.mul(self.act_func(self.gamma), -1)
                )
            )
        )

        return output
