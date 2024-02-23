import logging
import sys
import time
import os

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets.synapse_dataset import Synapse_dataset, RandomGenerator
from torch.utils.data import DataLoader
from utils.utils import DiceLoss


def trainer(model, max_epochs, base_lr, batch_size, train_path, dataset):
    checkpoint_path = f'./checkpoints/{model_name}_SGD_{base_lr}_{max_epochs}'
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

    print(f"The length of train set is: {len(data_train)}")

    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes = 9)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, weight_decay=0.0001, momentum=0.9)
    writer = SummaryWriter(checkpoint_path + '/log')
    iter_num = 0
    max_epoch = max_epochs
    max_iterations = max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations ")
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    start_time = time.time()
    all_epoch_time = max_iterations / 3600

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # image_batch, label_batch = image_batch.cuda().repeat(1,3,1,1), label_batch.cuda()
            image_batch, label_batch = image_batch.repeat(1,3,1,1) , label_batch
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('time(h): (%f/%f), epoch: (%d/%d), iteration: %d, loss: %f, loss_ce: %f, loss_dice: %f, lr: %f' % (((time.time() - start_time) * iter_num) / 3600, (time.time() - start_time) * all_epoch_time, epoch_num, max_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))
            start_time = time.time()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)


        if epoch_num >= max_epoch - 5:
            save_mode_path = os.path.join(checkpoint_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"