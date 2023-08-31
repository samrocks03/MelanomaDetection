from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
import cv2
import numpy as np
from PIL import Image
from time import perf_counter
import torch

class MelanomaDetection:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file("config.yaml")
        self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        self.cfg.MODEL.WEIGHTS = "model_final.pth"
        self.predictor = DefaultPredictor(self.cfg)
    
    def perform_melanoma_detection(self, file):

        s1 = perf_counter()

        # Processing Image
        self.file = file
        ima = Image.open(self.file)
        img = cv2.cvtColor(np.array(ima), cv2.COLOR_RGB2BGR)

        # Predicting
        op = self.predictor(img[...,::-1])
        
        # Outputs
        # label_mapping = {0: 'Benign', 1: 'Malignant'}
        # op["instances"].pred_classes = torch.Tensor([label_mapping[label] for label in op["instances"].pred_classes.tolist()])

        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale = 1.2)
        out = v.draw_instance_predictions(op["instances"].to("cpu"))
        output_image = out.get_image()[..., ::-1][..., ::-1]

        # Proper Output Processing
        d = {
            'imageName': file.name, 
            'imageWidth': op['instances'].image_size[0], 
            'imageHeight': op['instances'].image_size[1],
            'predictionDeviceType': str(self.cfg.MODEL.DEVICE)
        }
        details = []
        for i in range(len(op['instances'])):
            pred_class = op['instances'][i].pred_classes.item()
            if pred_class == 0:
                label = 'Benign'
            else:
                label = 'Malignant'
            details.append({'label': label, 
                            'confidence': str(round(100*op['instances'][i].scores.item(), 2)) + "%", 
                            'bbox': op['instances'][i].pred_boxes.tensor.tolist()})
        if len(details) == 0:
            d['objectsDetected'] = "Nothing Detected!"
        else:  
            d['objectsDetected'] = details 
        
        tt1 = str(round(perf_counter() - s1, 3)) + " seconds" 
        d['imageInferenceTime'] = tt1

        return d, output_image
    

        