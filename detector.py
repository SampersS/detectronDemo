from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy

class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        self.cfg.MODEL.DEVICE='cpu'
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.IMAGE_BW)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)
    


    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        if(cap.isOpened == False):
            print("Error opening the file...")
            return
        (succes, image) = cap.read()

        while succes:
            predictions = self.predictor(image)

            viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.IMAGE_BW)
            
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            cv2.imshow("Result", output.get_image()[:,:,::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (succes, image) = cap.read()