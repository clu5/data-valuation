import torch
import torch.nn as nn
from tqdm import tqdm


class embedding_model(torch.nn.Module):
    def __init__(
        self,
        model="clip",
    ):
        pass

    def forward(self):
        pass


def iteration(model, loader, optimizer, criterion, classification=False, train=True):
    """
    Perform a single iteration over a data loader.

    This function runs through the given data loader, applying the model, computing the loss,
    and updating the model if in training mode.

    Parameters:
        model (torch.nn.Module): The model to use.
        loader (torch.utils.data.DataLoader): The data loader for input data.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating the model.
        criterion (torch.nn.Module): The loss function to use.
        classification (bool, optional): Whether the task is classification or regression. Default is False.
        train (bool, optional): Whether to train or just evaluate the model. Default is True.

    Returns:
        float: The total loss over the iteration.
    """
    total_loss = 0
    model.train() if train else model.eval()
    # for x, y in tqdm(loader):
    for x, y in loader:
        x = x.cuda()
        if not classification:
            y = y.unsqueeze(1).float()
        y = y.cuda()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        if train:
            model.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().cpu().item()
    return total_loss


def fit(model, loader, optimizer, classification=True, val_loader=None, epochs=10):
    """
    Fit a model to the training data, with optional validation.

    This function trains a model over a specified number of epochs. It supports optional
    validation and handles both classification and regression tasks.

    Parameters:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating the model.
        classification (bool, optional): Whether the task is classification or regression. Default is True.
        val_loader (torch.utils.data.DataLoader, optional): The data loader for validation data. Default is None.
        epochs (int, optional): The number of training epochs. Default is 10.

    Returns:
        tuple or dict: If val_loader is provided, returns a tuple with training and validation loss dictionaries.
                       Otherwise, returns a dictionary with training loss only.
    """
    train_loss = {}
    val_loss = {}
    if classification:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    for i in tqdm(range(epochs)):
        train_loss[i] = iteration(
            model,
            loader,
            optimizer,
            criterion,
            classification=classification,
            train=True,
        )
        if val_loader is not None:
            val_loss[i] = iteration(
                model,
                val_loader,
                optimizer,
                criterion,
                classification=classification,
                train=False,
            )
    return (train_loss, val_loss) if val_loader else train_loss


class LinearRegression(nn.Module):
    """
    A simple linear regression model with one linear layer.
    
    This class represents a basic linear regression model with a single linear layer,
    designed for regression tasks.
    """
    def __init__(self, inp_dim=1000, out_dim=1):
        """
        Initialize the Linear Regression model.

        Parameters:
            inp_dim (int, optional): The number of input features. Default is 1000.
            out_dim (int, optional): The number of output features. Default is 1.
        """
        super().__init__()
        self.linear = nn.Linear(inp_dim, out_dim)

    def forward(self, x):
        """
        Forward pass of the linear regression model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear layer.
        """
        return self.linear(x)


class SoftmaxClassifier(nn.Module):
    """
    A simple softmax classifier with a linear layer.
    
    This class represents a basic softmax classifier with a single linear layer,
    designed for classification tasks.
    """
    def __init__(self, inp_dim=1000, out_dim=10):
        """
        Initialize the Softmax Classifier model.

        Parameters:
            inp_dim (int, optional): The number of input features. Default is 1000.
            out_dim (int, optional): The number of output features (number of classes). Default is 10.
        """
        super().__init__()
        self.linear = nn.Linear(inp_dim, out_dim)

    def forward(self, x):
        """
        Forward pass of the Softmax Classifier model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with softmax activation.
        """
        return torch.softmax(self.linear(x), dim=-1)


class CNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for classification or regression.
    
    This class represents a basic CNN with convolutional layers, pooling, and fully
    connected layers. It can be used for both classification and regression tasks.
    """
    def __init__(self, inp_channels=3, regressor=False):
        """
        Initialize the CNN model.

        Parameters:
            inp_channels (int, optional): The number of input channels (e.g., 3 for RGB images). Default is 3.
            regressor (bool, optional): Whether the model is for regression or classification. Default is False.
        """
        super().__init__()
        self.regressor = regressor
        self.conv1 = nn.Conv2d(inp_channels, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.relu = nn.functional.relu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(800, 500)
        self.fc1 = nn.Linear(1250, 500)
        self.fc2 = nn.Linear(500, 1 if regressor else 10)

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Parameters:
            x (torch.Tensor): Input tensor (e.g., images).

        Returns:
            torch.Tensor: Output tensor after passing through the CNN.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# class LeNet(Module):
# 	def __init__(self, numChannels, classes):
# 		# call the parent constructor
# 		super(LeNet, self).__init__()
# 		# initialize first set of CONV => RELU => POOL layers
# 		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
# 			kernel_size=(5, 5))
# 		self.relu1 = ReLU()
# 		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
# 		# initialize second set of CONV => RELU => POOL layers
# 		self.conv2 = Conv2d(in_channels=20, out_channels=50,
# 			kernel_size=(5, 5))
# 		self.relu2 = ReLU()
# 		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
# 		# initialize first (and only) set of FC => RELU layers
# 		self.fc1 = Linear(in_features=800, out_features=500)
# 		self.relu3 = ReLU()
# 		# initialize our softmax classifier
# 		self.fc2 = Linear(in_features=500, out_features=classes)
# 		self.logSoftmax = LogSoftmax(dim=1)
# 	def forward(self, x):
# 		# pass the input through our first set of CONV => RELU =>
# 		# POOL layers
# 		x = self.conv1(x)
# 		x = self.relu1(x)
# 		x = self.maxpool1(x)
# 		# pass the output from the previous layer through the second
# 		# set of CONV => RELU => POOL layers
# 		x = self.conv2(x)
# 		x = self.relu2(x)
# 		x = self.maxpool2(x)
# 		# flatten the output from the previous layer and pass it
# 		# through our only set of FC => RELU layers
# 		x = flatten(x, 1)
# 		x = self.fc1(x)
# 		x = self.relu3(x)
# 		# pass the output to our softmax classifier to get our output
# 		# predictions
# 		x = self.fc2(x)
# 		output = self.logSoftmax(x)
# 		# return the output predictions
# 		return output
