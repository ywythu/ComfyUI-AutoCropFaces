import os.path
from typing import List, Tuple, Any

import cv2
from insightface.app import FaceAnalysis
import numpy as np
import torch
import folder_paths
from face import Face
from log_init import logger

ImageNdarray = np.ndarray[Any, Any]


class InsightFaceBuffalo():
    """
    insightface buffalo_l 模型组带有如下五个模型：
        模型文件         大小   名称             用途                  代码调用实现
        --------------------------------------------------------------------------------------------------------
        det_10g.onnx    16MB  detection       用于脸部bbox和kps检测   @insightface.model_zoo.retinaface.py
        w600k_r50.onnx  166MB recognition     用于生成 embeddings    @insightface.model_zoo.arcface_onnx.py
        1k3d68.onnx     137MB landmark_3d_68  用于生成3d的68个关键点   @insightface.model_zoo.landmark.py
        ------------- 以下三个模型目前用不到 ---------------
        2d106det.onnx   4.8MB landmark_2d_106 用于生成2d的106个关键点  @insightface.model_zoo.landmark.py
        genderage.onnx  1.3MB genderage       用于推断性别、年龄       @insightface.model_zoo.attribute.py

    """
    face_detector: FaceAnalysis = None

    def __init__(self, execution_providers: List[str] = None):
        self.face_size = None
        probed_providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        self.execution_providers = execution_providers if execution_providers else probed_providers
        models_path = folder_paths.models_dir
        insightface_path = os.path.join(models_path, "insightface")
        self.face_detector = FaceAnalysis(name='buffalo_l', root=insightface_path,
                                                          providers=self.execution_providers)
        self.face_detector.prepare(ctx_id=0)

    def detect_faces(
            self,
            input_image: ImageNdarray,
            conf_threshold: float = 0.8,
            nms_threshold: float = 0.4,
            face_size: int = 512,
            crop_ratio: Tuple[int, int] = (1, 1),
            eye_dist_threshold: float = None,
    ) -> List[Face]:
        """
        retinaface 人脸检测模型返回的结果是一个 numpy.ndarray 浮点数数组的数组，每个元素数组为一张脸的数据
        元素数组长度为 15，例如 retinaface 模型
        元素 0-3 是脸部对应的 box，分别对应脸部方框的 startX, startY, endX, endY
        元素 4 是脸部置信度，通常为 0-1 的浮点数
        元素 5-15 为5个坐标点，5-6, 7-8, 9-10, 11-12, 13-14 分别对应 左瞳孔，右瞳孔，鼻尖，左嘴角，右嘴角 的 x,y 坐标

        Args:
            input_image: 输入图形
            conf_threshold: 最小置信度阈值
            nms_threshold: 未知阈值
            face_size: 用于标准化截取的脸部大小（正方形的边长像素值）
            crop_ratio: 用于标准化截取的放大大小
            eye_dist_threshold: 最小眼鼻距离的阈值
            only_largest_face: 是否只取最大的脸，缺省为True
            only_center_face: 是否只取最中间的脸，缺省为False

        Returns:
            `List[Face]`:
                Face 是一个 dict 其结构如下：
                {
                    "bbox":      人脸方框的起始、终止坐标数组 [start_x, start_y, end_x, end_y] 即 [left, top, right, bottom]，通过 detection 模型生成
                    "det_score": 人脸置信度，通过 detection 模型生成
                    "kps":       5个关键点坐标 [x, y] 的数组，分别对应 右眼瞳孔，左眼瞳孔，鼻尖，左嘴角，右嘴角 的 x,y 坐标，不同模型的定义可能不同，通过 detection 模型生成
                    "landmark_3d_68": 68个3d关键点，通过 landmark 模型生成，没有用到
                    "pose":      3个浮点数的数组，通过 landmark 模型生成，没有用到
                    "landmark_2d_106": 106个2d关键点，通过 landmark 模型生成，没有用到
                    "gender":    从头像推断出的性别 1 为男性，通过 genderage 模型生成，没有用到
                    "age":       从头像推断出的年龄，例如 44，通过 genderage 模型生成，没有用到
                    "embedding": 头像的embedding向量数据，通过 recognition 模型生成
                }
        """
        with torch.no_grad():
            faces = self.face_detector.get(input_image)
        face_count = len(faces)
        if face_count == 0:
            logger.warn(f'detected no faces')
            return []

        qualified_faces = []
        for idx, face in enumerate(faces):
            if face.det_score < conf_threshold:
                logger.info(
                    f'skipped face {idx} whose confidence {face.det_score} was less than conf_threshold {conf_threshold}')
                continue
            face_new = Face(face)
            qualified_faces.append(face_new)
        if len(qualified_faces) == 0:
            logger.warn(f'detected {len(qualified_faces)} qualified faces')
            return []

        face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                       [201.26117, 371.41043], [313.08905, 371.15118]])
        face_template = face_template * (face_size / 512.0)
        if crop_ratio[0] > 1:
            face_template[:, 1] += face_size * (crop_ratio[0] - 1) / 2
        if crop_ratio[1] > 1:
            face_template[:, 0] += face_size * (crop_ratio[1] - 1) / 2
        face_size = (int(face_size * crop_ratio[1]), int(face_size * crop_ratio[0]))

        for face in qualified_faces:
            # 有的图片检测出来会是坏脸
            # if is_bad_face(face, conf_threshold, eye_dist_threshold):
            #     continue
            # 根据脸部 kps 计算需要截取出来的脸部方框
            face.affine_matrix = cv2.estimateAffinePartial2D(face.kps, face_template, method=cv2.LMEDS)[0]
            face.cropped_face = cv2.warpAffine(
                input_image,
                face.affine_matrix,
                face_size,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(135, 133, 132))  # gray
            face.landmark_2d_68 = np.delete(face.landmark_3d_68, -1, axis=1)
        qualified_faces.sort(key=lambda x: (x.bbox[0], x.bbox[1]))
        return qualified_faces
