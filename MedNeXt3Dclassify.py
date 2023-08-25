import torch
import numpy as np
import SimpleITK as sitk
from medpy import metric
import torch.nn as nn
#import torchvision
# from torchstat import stat
import torchvision.models as models
import torchkeras
import torch.nn.functional as F


class MedNeXt_Block(nn.Module):
	def __init__(self, channels, ratio, kernel_size=(3, 3, 3), Groups=2):
		'''
		最基本的,输入输出不会变化的层

		一个倒置瓶颈层模块 包含一个K阶卷积层Depthwise Convolution Layer
		 一个Expansion Layer Layer 卷积核1X1X1
		 一个compreisson Convolution Layer 收缩层
		:param channels: 输入输出的层数
		:param ratio: 膨胀系数
		'''
		super(MedNeXt_Block, self).__init__()
		self.DWlayer = nn.Conv3d(channels, channels, kernel_size, stride=(1, 1, 1), padding=1)
		self.GNorm = nn.GroupNorm(Groups, channels)  # 这一步尚有欠缺
		self.Expansion_Layer = nn.Conv3d(channels, channels * ratio, kernel_size=1)
		self.GELU = nn.GELU()
		self.Compreisson_Layer = nn.Conv3d(channels * ratio, channels, kernel_size=1)

	def forward(self, x):
		residual = x
		x = self.DWlayer(x)
		x = self.GNorm(x)
		x = self.Expansion_Layer(x)
		x = self.GELU(x)
		x = self.Compreisson_Layer(x)
		x = x + residual
		return x


class MedNeXt_down_Block(nn.Module):
	def __init__(self, channels, ratio, kernel_size=(3, 3, 3), Groups=2,zdown=True):
		'''
		一个会下采样数据的模块
		一个倒置瓶颈层模块 用作下采样 减小特征尺寸，由普通的模块修改而来，注意减小尺寸和增加维度
		 包含一个K阶卷积层Depthwise Convolution Layer
		 一个Expansion Layer Layer 卷积核1X1X1
		 一个compreisson Convolution Layer 收缩层
		:param channels: 输入输出的层数
		:param ratio: 膨胀系数
		'''
		super(MedNeXt_down_Block, self).__init__()
		if zdown:
			self.DWlayer = nn.Conv3d(channels, channels, kernel_size, stride=(2, 2, 2), padding=1)
			self.residual_down = nn.Conv3d(channels, channels * 2, kernel_size=(1, 1, 1), stride=(2, 2, 2))
		else:
			self.DWlayer = nn.Conv3d(channels, channels, kernel_size, stride=(1, 2, 2), padding=1)
			self.residual_down = nn.Conv3d(channels, channels * 2, kernel_size=(1, 1, 1), stride=(1, 2, 2))
		self.GNorm = nn.GroupNorm(Groups, channels)  # 这一步尚有欠缺
		self.Expansion_Layer = nn.Conv3d(channels, channels * ratio, kernel_size=1)
		self.GELU = nn.GELU()
		self.Compreisson_Layer = nn.Conv3d(channels * ratio, channels * 2, kernel_size=1)


	def forward(self, x):
		residual = self.residual_down(x)
		x = self.DWlayer(x)
		x = self.GNorm(x)
		x = self.Expansion_Layer(x)
		x = self.GELU(x)
		x = self.Compreisson_Layer(x)
		x = x + residual
		return x


class Down_Stage(nn.Module):
	"""
	下采样stage 作为分类网络的话,差不多是 6个stage,下采样六次,,
	一个stage中通常由几个普通block加上一个下采样block组成
	这次如果不对输入的数据进行处理的话,可能会造成下采样的过程中数据不对的情况,对此
	"""
	def __init__(self, depth, channels, ratio,zdown=True):
		super(Down_Stage, self).__init__()
		self.blocks = nn.ModuleList()
		for i in range(depth):
			block = MedNeXt_Block(channels, ratio)
			self.blocks.append(block)
			pass
		self.down_block = MedNeXt_down_Block(channels, ratio,zdown=zdown)

	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		#short_cut = x
		x = self.down_block(x)
		return x

class MedNeXt_3D(nn.Module):
	'''
	一个由 MedNeXt组成的三维特征提取bottleneck
	'''
	def __init__(self, in_C=4, ratio=[2, 2, 2, 2, 2,2], block_list=[2, 2, 2, 2, 2,2],stage_depth=4):
		super(MedNeXt_3D, self).__init__()
		self.downsample_layers = nn.ModuleList()
		self.stem = nn.Conv3d(in_channels=1, out_channels=in_C, kernel_size=1, stride=1)
		self.down_stages = nn.ModuleList()
		self.num_stage=stage_depth
		for i in range(self.num_stage):
			if i%2==0:
				stage = Down_Stage(block_list[i], in_C * 2 ** (i), ratio[i])
			else:
				stage = Down_Stage(block_list[i], in_C * 2 ** (i), ratio[i],zdown=False)
			self.down_stages.append(stage)

	def forward(self, x):
		x = self.stem(x)
		for i in range(self.num_stage):
			x= self.down_stages[i](x)

		return x
		#这里的输出大概是不确定的,下面接上spp空间池化,筛选出固定的大小
	#最后的输出通道是 64(这个得算)




import math
class SPP3DLayer(torch.nn.Module):

	def __init__(self, num_levels=[1,2,4], pool_type='max_pool'):
		#将这个层改成三维的
		super(SPP3DLayer, self).__init__()

		self.num_levels = num_levels
		self.pool_type = pool_type

	def forward(self, x):
		num, c, d,h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
		for i in range(len(self.num_levels)):
			level = self.num_levels[i]
			kernel_size = (math.ceil(h / level), math.ceil(w / level))
			stride = (math.ceil(h / level), math.ceil(w / level))
			pooling = (
			math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

			h_wid = int(math.ceil(h / level))
			w_wid = int(math.ceil(w/level))
			d_wid = int(math.ceil(d/level))
			h_pad = math.floor((h_wid * level -h + 1) / 2)
			w_pad = math.floor((w_wid * level - w + 1) / 2)
			d_pad = math.floor((d_wid * level- d + 1) / 2)
			padding=(d_pad,h_pad,w_pad)

			# 选择池化方式
			if self.pool_type == 'max_pool':
				tensor = F.max_pool3d(x, kernel_size=(d_wid,h_wid,w_wid), stride=(d_wid,h_wid,w_wid),padding=padding)#.view(num, -1)
			else:
				tensor = F.avg_pool3d(x, kernel_size=(d_wid,h_wid,w_wid), stride=(d_wid,h_wid,w_wid),padding=padding)#.view(num, -1)
			# 展开、拼接
			if (i == 0):
				x_flatten = tensor.view(num, -1)
			else:
				x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
		return x_flatten#最终长度为[num,c*(73)]

class MLP(nn.Module):
	def __init__(self,num_input,num_hidden,num_output):
		super(MLP,self).__init__()
		self.fc1=nn.Linear(num_input,num_hidden)
		self.fc2=nn.Linear(num_hidden,num_hidden)
		self.GELU=nn.GELU()
		self.fc3=nn.Linear(num_hidden,num_output)

	def forward(self, x):
		x = self.GELU(self.fc1(x))  # 全连接层1 + ReLU激活函数
		x = self.GELU(self.fc2(x))  # 全连接层2 + ReLU激活函数
		x = self.fc3(x)  # 全连接层3（输出层）
		x = torch.squeeze(x, dim=1)
		x=torch.sigmoid(x)
		#print(x.shape,"模型输出的尺寸")
		return x

class MedNeXt_3D_sppClassifer(nn.Module):
	def __init__(self,in_C=4,nums_medstage=4,spp_list=[1,2,3]):
		super().__init__()
		self.MedNeXt=MedNeXt_3D(in_C=in_C,stage_depth=nums_medstage)
		self.sppnet=SPP3DLayer(num_levels=spp_list)
		sum=0
		for i in spp_list:
			sum+=i**3
		channelnum=in_C*2**(nums_medstage)
		self.MLP=MLP(sum*channelnum,2048,1)

	def forward(self,x):
		x=self.MedNeXt(x)
		x=self.sppnet(x)
		x=self.MLP(x)
		return x



if __name__ == "__main__":
	my_net = MedNeXt_3D_sppClassifer()
	torchkeras.summary(my_net, input_shape=(1, 42, 124, 124))
	pass