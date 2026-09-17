import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Feedfordward neural network for tabular classification.

    This architecture is created of intermediate lineal layers combined with
    non-linear activation functions (ReLU) to process input characteristics
    and predict class probability.

    :param input_dim: Feature number.
    :type input_dim: int
    :param num_classes: Total number of classes to predict.
    :type num_classes: int
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        """
        Executes fordwars pass of the neural network.

        It takes an input tensor batch and spread it over the sequential layers
        in order to generate non-normalized logits. 

        :param x: Input tensor with dimensions ``(batch_size, input_dim)``
        :type x: torch.Tensor
        :return: Output logits with dimensions ``(batch_size, num_classes)``
        :rtype: torch.Sensor
        """
        return self.network(x)