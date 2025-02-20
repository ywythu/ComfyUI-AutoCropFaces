import cv2
import numpy as np
from numpy.linalg import norm as l2norm


class Face(dict):
    """
    Face 是一个 dict 其结构如下：
    {
        "bbox":            人脸方框的起始、终止坐标数组 [start_x, start_y, end_x, end_y] 即 [left, top, right, bottom]
        "det_score":       人脸置信度,
        "kps":             5个关键点坐标 [x, y] 的数组，分别对应 右眼瞳孔，左眼瞳孔，鼻尖，左嘴角，右嘴角 的 x,y 坐标，不同模型的定义可能不同
        "affine_matrix":   5个关键点的仿射变换向量，形如 [[ 0.87498291 -0.41156097 16.6134035 ], [ 0.41156097 0.87498291 -188.795911 ]]
        "cropped_face":    根据bbox和关键点仿射变换向量截取出的脸部小图
        "landmark_3d_68":  68个3d关键点的ndarray，shape为 (68, 3)
        "landmark_2d_68":  68个2d关键点的ndarray，shape为 (68, 2)
        "pose":            3个浮点数的数组
        "landmark_2d_106": 106个2d关键点
        "gender":          从头像推断出的性别 1 为男性
        "age":             从头像推断出的年龄，例如 44
        "embedding":       头像的embedding向量数据
    }
    """

    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        # for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'
