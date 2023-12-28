from mmengine.runner.loops import TestLoop
from mmyolo.registry import LOOPS
from typing import Dict, List, Union
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger


@LOOPS.register_module()
class GeneralityTestLoop(TestLoop):
    """Loop for generality test.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 corruptions=None,
                 severities=None,
                 gen_type='custom',  # 'custom' 为自定义方法生成测试集 'standard' 为调用imagecorruptions库生成测试集
                 fp16: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        if corruptions is None:
            if gen_type is 'custom':
                self.corruptions = ['guassian_noise',
                                    'shot_noise',
                                    'impulse_noise',
                                    'defocus_blur',
                                    'frosted_glass_blur',
                                    'motion_blur',
                                    'zoom_blur',
                                    'snow',
                                    'rain',
                                    'fog',
                                    'brightness',
                                    'contrast',
                                    'elastic',
                                    'pixelate',
                                    'jpeg']
            else:
                self.corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                     'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                     'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                     'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
        else:
            self.corruptions = corruptions
        if severities is None:
            self.severities = [1, 2, 3, 4, 5]
        else:
            self.severities = severities
        self.gen_prefix = 'corruptions' if gen_type is 'custom' else 'corruptions1'

    def run(self) -> dict:
        """Launch test."""
        ori_metrics = super().run()
        logger: MMLogger = MMLogger.get_current_instance()
        corruption_dataloader_cfg = self.runner.cfg.get('test_dataloader').copy()
        log_metrics = dict()
        sum_cor_metrics = dict()
        sum_metrics = dict()
        for cor in self.corruptions:
            # 记录初始值
            for (key, value) in ori_metrics.items():
                log_metrics[key.replace('coco', cor)] = value
            self.runner.visualizer.add_scalars(log_metrics)
            sum_cor_metrics.clear()

            for ser in self.severities:
                logger.info('Testing corruption {0} at severity {1}'.format(cor, ser))
                corruption_dataloader_cfg.get('dataset').get('data_prefix')['img'] = ('{0}/{1}/{2}'
                                                                                      .format(self.gen_prefix, cor, ser))
                corruption_dataloader = self.runner.build_dataloader(corruption_dataloader_cfg)
                self.runner.call_hook('before_test_epoch')
                for idx, data_batch in enumerate(corruption_dataloader):
                    self.run_iter(idx, data_batch)
                metrics = self.evaluator.evaluate(len(corruption_dataloader.dataset))

                for (key, value) in metrics.items():
                    log_key = key.replace('coco', cor)
                    if log_key in sum_cor_metrics:
                        sum_cor_metrics[log_key] += metrics[key] / len(self.severities)
                    else:
                        sum_cor_metrics[log_key] = metrics[key] / len(self.severities)
                    log_metrics[log_key] = value
                # 记录
                self.runner.visualizer.add_scalars(log_metrics)
                log_metrics.clear()

            for key in sum_cor_metrics.keys():
                k = key.split('/')[1]
                if k in sum_metrics:
                    sum_metrics[k] += sum_cor_metrics[key] / len(self.corruptions)
                else:
                    sum_metrics[k] = sum_cor_metrics[key] / len(self.corruptions)

            logger.info('metrics in corruption {0} is: \n'.format(cor))
            for (k,v) in sum_cor_metrics.items():
                logger.info('{0}\t\t:{1:.3f}'.format(k, v))
            log_metrics['metrics_mAP/{0}'.format(cor)] = sum_cor_metrics['{0}/bbox_mAP'.format(cor)]
            self.runner.visualizer.add_scalars(log_metrics)
            log_metrics.clear()

        logger.info('metrics in total is: \n')
        for (k, v) in sum_metrics.items():
            logger.info('{0}\t:{1:.3f}'.format(k, v))
        log_metrics['metrics_mAP/total'] = sum_cor_metrics['bbox_mAP']
        self.runner.visualizer.add_scalars(log_metrics)
        log_metrics.clear()

        return sum_metrics
