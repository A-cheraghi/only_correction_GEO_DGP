import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'])

    train_batches, val_batches = 0 , 0
    train_batches, val_batches = prepare_batched_cached_data(cfg['dataset'])
    # val_batches= prepare_batched_cached_data(cfg['dataset'])


# # نمایش داده ها
#     import numpy as np

#     # فرض می‌کنیم train_batches ساخته شده است
#     first_batch = val_batches[1]  # گرفتن اولین بچ
#     sample_in_batch = 0           # نمایه/اندیس نمونه مورد نظر درون این بچ

#     def get_sample(tensor):
#         """تابع کمکی برای استخراج مقدار نمونه اول فارغ از ابعاد تانسور"""
#         if tensor.ndim == 4:    # [Layers, Batch_Size, Queries, Dim]
#             return tensor[-1, sample_in_batch, 0, :5].detach().cpu().numpy()
#         elif tensor.ndim == 3:  # [Batch_Size, Queries, Dim]
#             return tensor[sample_in_batch, 0, :5].detach().cpu().numpy()
#         elif tensor.ndim == 2:  # [Batch_Size, Dim]
#             return tensor[sample_in_batch, :5].detach().cpu().numpy()
#         else:
#             return tensor[sample_in_batch].flatten()[:5].detach().cpu().numpy()

#     log_text = (
#         "\n" + "="*70 + "\n"
#         f">>> BATCHED CACHED DATA DEBUG (Sample {sample_in_batch} in Batch 0) <<<\n"
#         + "="*70 + "\n"
#         f"🔹 outputs_coord (Layer Last, Q0, First 4) : {get_sample(first_batch['outputs_coord'])[:4]}\n"
#         f"🔹 outputs_coord_logits (Last, Q0, First 4): {get_sample(first_batch['outputs_coord_logits'])[:4]}\n"
#         f"🔹 outputs_class (Layer Last, Q0, First 3)  : {get_sample(first_batch['outputs_class'])[:3]}\n"
#         f"🔹 outputs_3d_dim (Layer Last, Q0, All 3)  : {get_sample(first_batch['outputs_3d_dim'])[:3]}\n"
#         f"🔹 outputs_depth (Layer Last, Q0, All 2)  : {get_sample(first_batch['outputs_depth'])[:2]}\n"
#         f"🔹 outputs_angle (Layer Last, Q0, All 2)  : {get_sample(first_batch['outputs_angle'])[:2]}\n"
#         f"🔹 inter_class (Layer Last, Q0, First 3)   : {get_sample(first_batch['inter_class'])[:3]}\n"
#         f"🔹 inter_coord (Layer Last, Q0, First 4)   : {get_sample(first_batch['inter_coord'])[:4]}\n"
#         f"🔹 hs_2d_last (Q0, First 5)               : {first_batch['hs_2d_last'][sample_in_batch, 0, :5].detach().cpu().numpy()}\n"
#         f"🔹 hs_3d_last (Q0, First 5)               : {first_batch['hs_3d_last'][sample_in_batch, 0, :5].detach().cpu().numpy()}\n"
#         f"🔹 pred_depth_map_logits (First 5)        : {first_batch['pred_depth_map_logits'][sample_in_batch].flatten()[:5].detach().cpu().numpy()}\n"
#     )

#     # اضافه کردن لیست region_probs (شامل ۴ لایه)
#     log_text += "🔹 region_probs (First 4 values per layer):\n"
#     for idx, r_prob in enumerate(first_batch['region_probs']):
#         log_text += f"    └─ Layer {idx}: {r_prob[sample_in_batch].flatten()[:4].detach().cpu().numpy()}\n"

#     log_text += "="*70 + "\n"

#     # اگر خواستی درون فایل متنی هم ذخیره بشود:
#     with open("/content/batched_debug.txt", "a") as f:
#         f.write(log_text)




    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name,
                        val_batches=val_batches)
        tester.test()
        return
    #ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name,
                      train_batches=train_batches)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name,
                    val_batches=val_batches)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()



# --- ۱. تعریف تابع ساخت بچ‌ها (خارج از main) ---
def prepare_batched_cached_data(cfg):
    """
    فایل‌های کش‌شده train و val را می‌خواند و هر دو را به صورت بچ‌بندی شده برمی‌گرداند.
    """
    root_dir = cfg['root_dir']
    batch_size = cfg['batch_size']
    
    # لیست کلیدهای ۴بعدی که بعد نمونه‌ها در آن‌ها dim=1 است
    layer_tensors = [
        "outputs_coord", "outputs_coord_logits", "outputs_class", 
        "outputs_3d_dim", "outputs_depth", "outputs_angle", 
        "inter_class", "inter_coord"
    ]

    def create_batches_for_split(split_name):
        file_name = f"cached_features_{split_name}_unified.pt"
        cached_data_path = os.path.join(root_dir, file_name)
        
        cached_data = torch.load(cached_data_path, map_location="cpu")
        
        total_samples = cached_data["hs_2d_last"].shape[0]
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        batched_list = []
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, total_samples)
            
            batch_dict = {}
            for key, val in cached_data.items():
                if key == "region_probs":
                    # اسلایس زدن روی بعد بچ برای تک‌تک 4 لایه داخل لیست region_probs
                    batch_dict[key] = [layer_tensor[start:end] for layer_tensor in val]
                elif key in layer_tensors:
                    batch_dict[key] = val[:, start:end]
                else:
                    # شامل pred_depth_map_logits، hs_2d_last و سایر تانسورهای معمولی
                    batch_dict[key] = val[start:end]
                    
            batched_list.append(batch_dict)
            
        return batched_list

    # ساخت بچ‌ها برای هر دو مجموعه داده
    # train_batches = create_batches_for_split('train')
    val_batches = create_batches_for_split('val')

    # return train_batches, val_batches
    return val_batches


if __name__ == '__main__':
    main()

