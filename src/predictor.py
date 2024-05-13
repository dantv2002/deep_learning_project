import os

import cv2
import gdown
import torch

# detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


class Visualization:
    def __init__(self):
        # download model detectron2
        path_model_detectron2 = "../models/detectron2"
        if not os.path.exists(path_model_detectron2):
            os.makedirs(path_model_detectron2)
        self.download_model(
            output_dir=path_model_detectron2
        )  # dowload transfer learning model zoo R50_FPN_3x
        self.download_model(
            output_dir=path_model_detectron2,
            file_name="/Model_R50_C4_3x.pth",
            url="https://drive.google.com/uc?id=15VYtOlXyUqkkjrBx9IfDiCbP-796izJY",
        )  # dowload transfer learning model zoo R50_C4_3x
        #
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
        self.fruit_metadata = {"thing_classes": class_names}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_list = (
            []
        )  # detectron2 với chỉ mục 0: model Model_R50_FPN_3x, chỉ mục 1: model Model_R50_C4_3x
        self.model_selected = 0
        self.detectron2_create_model(
            device=device, num_classes=len(class_names)
        )  # Tạo model 1
        self.detectron2_create_model(
            device=device,
            cfg_model_zoo="COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
            path_weight="../models/detectron2/Model_R50_C4_3x.pth",
            score=0.7,
        )

    def cfg(self, model):
        print("Configuring model...")
        print(f"Model: {model}")
        self.model_selected = model - 1

    def run(self, images) -> list:
        print("Running...")
        batch_image = [cv2.imread(image["image_path"]) for image in images]
        image_result = None
        if self.model_selected == 0 or self.model_selected == 1:
            image_result = self.detectron2_predict(batch_image)
        results = []
        for i in range(len(images)):
            if image_result is not None:
                results.append(
                    {"id": images[i]["id"], "image": image_result[i]}
                )
        print("Done!")
        return results

    def detectron2_predict(self, batch_image):
        print("Predicting...")
        results = []
        predictor = self.predictor_list[self.model_selected]
        for image in batch_image:
            outputs = predictor(image)["instances"].to("cpu")
            results.append(self.detectron2_visualizer(image, outputs))
        return results

    def detectron2_visualizer(self, img, outputs):
        print("Visualizer...")
        v = Visualizer(
            img[:, :, ::-1], metadata=self.fruit_metadata, font_size_scale=1.5
        )
        image = v.draw_instance_predictions(outputs).get_image()
        return image

    def detectron2_create_model(
        self,
        cfg_model_zoo="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        path_weight="../models/detectron2/Model_R50_FPN_3x.pth",
        device="cpu",
        score=0.62,
        num_classes=20,
    ):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = device
        cfg.merge_from_file(model_zoo.get_config_file(cfg_model_zoo))
        cfg.MODEL.WEIGHTS = path_weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.predictor_list.append(DefaultPredictor(cfg))

    def download_model(
        self,
        output_dir="../models/detectron2",
        file_name="/Model_R50_FPN_3x.pth",
        url="https://drive.google.com/uc?id=13n1JNlhku8FxYjTwSvm-BIkQTwxSrm6b",
    ):
        if not os.path.exists(output_dir + file_name):
            gdown.download(
                url=url,
                output=output_dir + file_name,
                quiet=True,
            )
