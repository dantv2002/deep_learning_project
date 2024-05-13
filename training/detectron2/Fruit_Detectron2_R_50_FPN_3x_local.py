"""
    Thực hiện: Huỳnh Thanh Tuấn
    Mã số sinh viên: 20110120
"""

# %%
# NOTE Thêm thư viện
import os

import cv2
import matplotlib.pyplot as plt
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

# %%
# Chạy pretrained model trên ảnh test
# NOTE (OPTIONAL)
img = cv2.imread(
    "./dataset/test/AnhDao-106-_jpg.rf.bceaa204d8ae714a07028dd4e8dcabb8.jpg"
)
img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_show)
plt.axis("off")
plt.show()

# %%
# NOTE (OPTIONAL)
config = get_cfg()
config.MODEL.DEVICE = "cpu"
config.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Đặt ngưỡng điểm số nhận diện
config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)
predictor_sample = DefaultPredictor(config)
outputs = predictor_sample(img)["instances"]
print(outputs.pred_classes)
print(outputs.pred_boxes)

# %%
# NOTE (OPTIONAL)
v = Visualizer(
    img[:, :, ::-1],  # chuyển từ BGR (opencv) sang RGB
    MetadataCatalog.get(config.DATASETS.TRAIN[0]),
    scale=1.5,
)
out = v.draw_instance_predictions(outputs)
plt.imshow(out.get_image())
plt.axis("off")
plt.show()

# %%
# NOTE Bước 1: Chuẩn bị dữ liệu
class_names = [
    "Trai-Cay",
    "Anh-Dao",
    "Bo",
    "Buoi",
    "Cam-Sanh",
    "Cam-Vang",
    "Chanh",
    "Coc",
    "Dau",
    "Du-Du",
    "Dua-Hau",
    "Dua-Luoi",
    "Hong",
    "Khe",
    "Kiwi",
    "Le",
    "Mang-Cau",
    "Mit",
    "Tao",
    "Viet-Quat",
    "Xoai",
]
path_train = "./dataset/train"
path_valid = "./dataset/valid"
path_test = "./dataset/test"
# Đăng ký tập dữ liệu Train, Valid và Test
register_coco_instances(
    "fruit_train", {}, path_train + "/_annotations.coco.json", path_train
)
register_coco_instances(
    "fruit_valid", {}, path_valid + "/_annotations.coco.json", path_valid
)
register_coco_instances(
    "fruit_test", {}, path_test + "/_annotations.coco.json", path_test
)

# %%
# NOTE Bước 2: Xây dựng model transfer learning với model zoo COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
device = "cuda" if torch.cuda.is_available() else "cpu"


config = get_cfg()
config.MODEL.DEVICE = device
config.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)

config.DATASETS.TRAIN = ("fruit_train",)
config.DATASETS.TEST = ("fruit_valid",)

config.DATALOADER.NUM_WORKERS = 0  # Số luồng(threads) tải dữ liệu
config.SOLVER.IMS_PER_BATCH = 6  # Batch size

# NOTE giá trị mặc định của config: https://detectron2.readthedocs.io/en/latest/modules/config.html
# config.OUTPUT_DIR=output
# config.SOLVER.BASE_LR_END=0.0
# config.SOLVER.WARMUP_FACTOR=1.0 / 1000
# config.MODEL.BACKBONE.FREEZE_AT=2 (đóng băng 2 stage đầu của mạng ResNet-50)

n_epochs = 18  # số lượng epoch

iter_per_epoch = int(
    len(DatasetCatalog.get("fruit_train")) / config.SOLVER.IMS_PER_BATCH
)
max_iters = iter_per_epoch * n_epochs

config.SOLVER.MAX_ITER = max_iters

config.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
config.SOLVER.BASE_LR = 0.01  # Đặt tốc độ học tối đa
config.SOLVER.WARMUP_ITERS = int(0.2 * n_epochs * iter_per_epoch)

config.TEST.EVAL_PERIOD = iter_per_epoch

config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128  # Số lượng vùng quan tâm (ROIS) được lấy tối đa từ mỗi hình ảnh
)
config.MODEL.ROI_HEADS.NUM_CLASSES = len(
    class_names
)  # Số lượng lớp cần phân loại

os.makedirs(
    config.OUTPUT_DIR, exist_ok=True
)  # Tạo thư mục ouput (giá trị của config.OUTPUT_DIR)
os.makedirs(
    config.OUTPUT_DIR + "/inference", exist_ok=True
)  # Tạo thư mục inference chứa kết quả evaluator khi training


# %%
# NOTE Bước 3: Training model
class MyTrainer(DefaultTrainer):
    # Override
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = config.OUTPUT_DIR + "/inference"
        return COCOEvaluator(
            dataset_name, ["bbox"], False, output_dir=output_folder
        )

    # Override
    def build_hooks(self):
        checkpointer = DetectionCheckpointer(
            self.model, save_dir=config.OUTPUT_DIR
        )  # Tạo đối tượng checkpointer để lưu model_best
        hooks = super().build_hooks()
        best_checkpointer = BestCheckpointer(
            eval_period=iter_per_epoch,
            checkpointer=checkpointer,
            val_metric="bbox/AP",
        )
        hooks.append(best_checkpointer)
        return hooks


trainer = MyTrainer(config)
trainer.resume_or_load(resume=False)

# %%
# NOTE: Chạy training
trainer.train()

# %%
# NOTE Bước 4: Chạy model_best trên ảnh test
# Tải model
cfg = get_cfg()
cfg.MODEL.DEVICE = device
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Đặt ngưỡng điểm số nhận diện
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

predictor = DefaultPredictor(cfg)

image_sample = (
    "./dataset/test/AnhDao-106-_jpg.rf.bceaa204d8ae714a07028dd4e8dcabb8.jpg"
)
img = cv2.imread(image_sample)
outputs = predictor(img)["instances"].to(
    "cpu"
)  # chuyển kết quả dự đoán từ gpu sang cpu nếu đang chạy trên device cuda
print(outputs.pred_classes)
print(outputs.pred_boxes)

# %%
# NOTE: Sử dụng Visualizer vẽ kết quả lên ảnh
fruit_metadata = {"thing_classes": class_names}
v = Visualizer(img[:, :, ::-1], metadata=fruit_metadata, font_size_scale=1.5)
out = v.draw_instance_predictions(outputs)
plt.imshow(out.get_image())
plt.axis("off")
plt.show()

# %%
# NOTE Bước 5: Đánh giá model
evaluator = COCOEvaluator(
    "fruit_test", output_dir=cfg.OUTPUT_DIR
)  # Tạo đối tượng đánh giá
test_loader = build_detection_test_loader(
    cfg, "fruit_test"
)  # Tạo đối tượng tải dữ liệu test
print(
    inference_on_dataset(predictor.model, test_loader, evaluator)
)  # In ra kết quả chạy model trên test_loader(tập dữ liệu test) và đánh giá kết quả bằng evaluator

# %%
