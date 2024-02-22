import logging
import sys

from torchvision import transforms
import torch.nn as nn

from datasets.synapse_dataset import Synapse_dataset, RandomGenerator
from torch.utils.data import DataLoader

model_name = 'ParaTransCNN'
max_epochs = 150
base_lr = 0.01
batch_size = 4
train_path = '/data/Synapse/train/'
dataset = 'synapse'


checkpoint_path = './checkpoints/{}_SGD_{}_{}'.format(model_name, base_lr, max_epochs)

def trainer(model, max_epochs, base_lr, batch_size, train_path, dataset, checkpoint_path):
    logging.basicConfig(filename=checkpoint_path+'/log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info('Training started')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model_name = 'ParaTransCNN'
    max_epochs = 150
    base_lr = 0.01
    batch_size = 8
    train_path = '/data/Synapse/train/'
    dataset = 'synapse'
    list_dir = '/lists/lists_Synapse/'
    img_size = 224


    data_train = Synapse_dataset(base_dir=train_path, list_dir=list_dir, split="train",
                        transform=transforms.Compose(
                            [RandomGenerator(output_size=[img_size, img_size])]))

    print("The length of train set is: {}".format(len(data_train)))

    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    
    ce_loss = nn.CrossEntropyLoss()