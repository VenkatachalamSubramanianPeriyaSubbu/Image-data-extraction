import pytesseract
import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO

def get_bounding_box(image_array):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'yolov5v8.pt')
    # model = YOLO('yolov8x-finetuned.pt')
    results = model(image_array)
    data = results.pandas().xyxy[0]
    # data = []
    # for result in results:  # YOLOv8 may return multiple results for batch inference
    #     for box in result.boxes:  # Access each detected box
    #         box_data = {
    #             'xmin': box.xyxy[0][0].item(),
    #             'ymin': box.xyxy[0][1].item(),
    #             'xmax': box.xyxy[0][2].item(),
    #             'ymax': box.xyxy[0][3].item(),
    #             'confidence': box.conf.item(),
    #             'class': int(box.cls.item()),
    #             'name': result.names[int(box.cls.item())]
    #         }
    #         data.append(box_data)
    return pd.DataFrame(data), results

def get_box_coordinates(datadf):
    main_dict = {}
    for i in range(datadf.shape[0]):
        subdict = {}
        subdict['xmin'] = datadf['xmin'][i]
        subdict['ymin'] = datadf['ymin'][i]
        subdict['xmax'] = datadf['xmax'][i]
        subdict['ymax'] = datadf['ymax'][i]
        subdict['confidence'] = datadf['confidence'][i]
        main_dict[datadf['name'][i]] = subdict
    return main_dict

def get_coordinates(image, data, obj, delta=0):
    try:
        cropped_image = image[int(data[obj]['ymin']-delta):int(data[obj]['ymax']+delta),
                            int(data[obj]['xmin']-delta):int(data[obj]['xmax']+delta)]
        text = pytesseract.image_to_string(cropped_image)
    except Exception as e:
        return f'Error extracting text from {obj} OR {obj} not found : {e}'
    return text.split()
