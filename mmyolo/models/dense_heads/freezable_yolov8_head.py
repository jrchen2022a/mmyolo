from mmyolo.registry import MODELS
from .yolov8_head import YOLOv8Head


@MODELS.register_module()
class FreezableYOLOv8Head(YOLOv8Head):

    def __init__(self,
                 frozen_head:bool=False,
                 **kwargs):
        self.frozen_head = frozen_head
        super().__init__(**kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen_head:
            for param in self.parameters():
                param.requires_grad = False
