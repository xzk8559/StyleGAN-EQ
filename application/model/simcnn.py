import torch
import torch.nn as nn
import torch.nn.functional as F 

class simCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( 1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.prlu1 = nn.PReLU(16)
        self.prlu2 = nn.PReLU(32)
        self.prlu3 = nn.PReLU(32)
        self.prlu4 = nn.PReLU(64)
        self.prlu5 = nn.PReLU(64)
        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.constant_(self.conv1.bias, 0)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.constant_(self.conv2.bias, 0)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.constant_(self.conv3.bias, 0)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.constant_(self.conv4.bias, 0)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.constant_(self.conv5.bias, 0)
        
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 40*3)

    def forward(self, x):
        
        x = self.prlu1((self.conv1(x)))
        x = self.prlu2((self.conv2(x)))
        x = self.prlu3((self.conv3(x)))
        x = self.prlu4((self.conv4(x)))
        x = self.prlu5((self.conv5(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
class SEModule(nn.Module):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x
    