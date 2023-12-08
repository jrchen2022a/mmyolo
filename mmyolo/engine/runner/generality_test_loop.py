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
                 fp16: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        if corruptions is None:
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
            self.corruptions = corruptions
        if severities is None:
            self.severities = [1, 2, 3, 4, 5]
        else:
            self.severities = severities

    def run(self) -> dict:
        """Launch test."""
        # ret_metrics = super().run()
        logger: MMLogger = MMLogger.get_current_instance()
        corruption_dataloader_cfg = self.runner.cfg.get('test_dataloader').copy()

        sum_metrics = dict()
        for cor in self.corruptions:
            for ser in self.severities:
                logger.info('Testing corruption {0} at severity {1}'.format(cor, ser))
                corruption_dataloader_cfg.get('dataset').get('data_prefix')['img'] = ('corruptions/{0}/{1}/'
                                                                                      .format(cor, ser))
                corruption_dataloader = self.runner.build_dataloader(corruption_dataloader_cfg)
                self.runner.call_hook('before_test_epoch')
                for idx, data_batch in enumerate(corruption_dataloader):
                    self.run_iter(idx, data_batch)
                metrics = self.evaluator.evaluate(len(corruption_dataloader.dataset))
                for (key, value) in metrics.items():
                    if key in sum_metrics:
                        sum_metrics[key] += metrics[key]
                    else:
                        sum_metrics[key] = metrics[key]

            for k in sum_metrics.keys():
                sum_metrics[k] /= len(self.severities)
            logger.info('metrics in corruption {0} is: \n'.format(cor))
            for (k,v) in sum_metrics.items():
                logger.info('{0}\t:{1:.3f}'.format(k, v))

        for k in sum_metrics.keys():
            sum_metrics[k] /= len(self.corruptions)
        logger.info('metrics in total is: \n')
        for (k, v) in sum_metrics.items():
            logger.info('{0}\t:{1:.3f}'.format(k, v))
        # return ret_metrics
        return None
