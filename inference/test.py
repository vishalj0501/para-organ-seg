import logging

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from utils.utils import DiceLoss, test_single_volume



def inference_Synapse(model, test_save_path=None):
    db_test = Synapse_dataset(base_dir="data/Synapse/test_vol", split="test_vol", list_dir="lists/lists_Synapse/test_vol.txt")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()
    metric_list = 0.0
    num_classes = 9 

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=num_classes, patch_size=[224, 224],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice : %f mean_hd95 : %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)

    for i in range(1, num_classes):
        logging.info('Mean class %d mean_dice : %f mean_hd95 : %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"
