# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng Yutao
@Time    : 2019/11/28 9:33
"""
import logging, os, torch
from Timer import *
from _02PipeDatasetLoader import *
from _03Unet import *
from _04Loss import *

WeightCoefficient = 2
Lr = 0.01
Epochs = 700
LrDecay = 0.1
BatchSize = 25
LrDecayPerEpoch = 500  # 学习率调整的epoch
ValidPerEpoch = 50  # 测试的epoch
SavePerEpoch = 700  # 保存结果的epoch
torch.cuda.set_device(0)  # 选用GPU设备

# %% TODO:载入数据，初始化网络，定义目标函数
FolderPath = '../Dataset'
TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, BatchSize, False)
# %% Unet_BCELoss_Adam
Unet = UNet(in_channels=3, out_channels=1, init_features=4, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to('cuda')
SaveFolder = 'Output'
Criterion = nn.BCELoss().to('cuda')
Optimizer = torch.optim.Adam(Unet.parameters(), lr=Lr)
os.makedirs(SaveFolder, exist_ok=SaveFolder)
logging.basicConfig(filename=os.path.join(SaveFolder, 'log.txt'), filemode='w', level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')
# Unet.load_state_dict(torch.load(os.path.join(SaveFolder, 'PreTrained.pt'), map_location = 'cuda'))
logging.warning('WeightCoefficient:{0:03d}'.format(WeightCoefficient))

# %% TODO:开始循环训练
LrScheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=LrDecayPerEpoch, gamma=LrDecay)  # 设置学习率策略
for Epoch in range(1, Epochs + 1):
	# %% 训练
	Unet.train()  # 训练模式
	# torch.cuda.empty_cache()  # 释放缓存占用, 耗时大概0.05s
	# 训练一个Epoch
	TrainLoss = 0
	print('Epoch:%d, LR:%.8f ' % (Epoch, LrScheduler.get_lr()[0]), end='>> ', flush=True)
	for Iter, (InputImg, Label, SampleName) in enumerate(TrainDataLoader):
		print(Iter, end=' ', flush=True)
		InputImg = InputImg.float().to('cuda')
		Label = Label.float().to('cuda')
		Weight = Label * (WeightCoefficient-1) + 1
		Criterion.weight = Weight
		Optimizer.zero_grad()
		with torch.set_grad_enabled(True):
			OutputImg = Unet(InputImg)
			BatchLoss = Criterion(OutputImg, Label)
			BatchLoss.backward()
			Optimizer.step()
			TrainLoss += BatchLoss.item()
			# print(OutputImg.shape)
	AveTrainLoss = TrainLoss / TrainDataset.__len__() * BatchSize  # 平均每幅图像的loss
	print(", Total loss is: %.6f" % float(AveTrainLoss))
	logging.warning('\tTrain\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}'.format(Epoch, LrScheduler.get_lr()[0], AveTrainLoss))

	# %% 测试
	if Epoch % ValidPerEpoch == 0 or Epoch == 1:
		Unet.eval()  # 训练模式
		torch.cuda.empty_cache()  # 释放缓存占用
		ValLoss = 0
		print('Validate:', end='>>', flush=True)
		for Iter, (InputImg, Label, SampleName) in enumerate(ValDataLoader):
			print(Iter, end=' ', flush=True)
			InputImg = InputImg.float().to('cuda')
			Label = Label.float().to('cuda')
			Weight = Label * (WeightCoefficient - 1) + 1
			Criterion.weight = Weight
			with torch.set_grad_enabled(False):  # 等同于torch.no_grad()
				OutputImg = Unet(InputImg)
				BatchLoss = Criterion(OutputImg, Label)  # CrossEntropyLoss的Target必须没有通道的维度，即(BatchSize, W, H)
				ValLoss += BatchLoss.item()

				# pred = np.array(OutputImg.data.cpu()[0])[0]
				# pred = (Normalization(pred) * 255).astype(np.uint8)
				# pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
				# plt.imshow(pred)
				# plt.show()

		AveValLoss = ValLoss / ValDataset.__len__()
		print("Total loss is: %.6f" % float(AveValLoss))
		logging.warning('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tValid\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}'.format(Epoch, LrScheduler.get_lr()[0], AveValLoss))

	# %% 保存
	if Epoch % SavePerEpoch == 0:
		torch.save(Unet.state_dict(), os.path.join(SaveFolder, '{0:04d}_dict.pt'.format(Epoch)))
		torch.save(Unet, os.path.join(SaveFolder, '{0:04d}.pt'.format(Epoch)))

	# %% 每隔一定epoch后更新一次学习率
	LrScheduler.step()
