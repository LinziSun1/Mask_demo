import pandas as pd
import cv2
import numpy as np

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response

def model_prepared(model_path):
    device = torch.device("cpu")
    model = torch.load(model_path)
    #model.eval
    return model

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# def get_image(img_path):
#     img = Image.open(img_path) # Load the image
#     transform = transforms.Compose([transforms.ToTensor()])
#     img = transform(img) # Apply the transform to the image
#     return img

def get_prediction(model,img_path):
    img = get_image(img_path)
    model.eval()
    device = torch.device("cpu")
    pred = model([img.to(device)])
    pred = [{k: v.detach().cpu() for k, v in detection.items()} for detection in pred]
    return pred

def plot_face_mask_scores(img, annotation):
    
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    
    for box, label,scores in zip(annotation["boxes"], annotation["labels"],annotation["scores"]):
        xmin, ymin, xmax, ymax = box
        score = round(scores.item(),2)
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.axis('off')
        if label == 1:
            ax.text(xmin, ymin, "Mask:"+str(score), color='g')
        elif label == 2:
            ax.text(xmax, ymax, "Incorrect Mask:"+str(score), color='y')
        else:
            ax.text(xmin, ymin, "No Mask:"+str(score), color='r')

    #plt.show()
    return fig


# def main():
#     model = get_model_instance_segmentation(3)
#     model.load_state_dict(torch.load('D:\Desktop\Client/model_fasterrcnn_resnet50_fpn.pt', map_location=torch.device('cpu')))
#     model.eval()
#     #detect_objects_on_camera()
#     user_input = input("Please enter the image path: ")
#     img_path = user_input
#     pred = get_prediction(model,img_path)
#     plot_face_mask_scores(img_path, pred[0])

# def process_image(image):
#     model = get_model_instance_segmentation(3)  
#     model.load_state_dict(torch.load('D:\Desktop\Client/model_fasterrcnn_resnet50_fpn.pt', map_location=torch.device('cpu')))   
#     model.eval()
#     #img = Image.open(image)
#     transform = transforms.Compose([transforms.ToTensor()])
#     img = transform(image)
#     device = torch.device("cpu")
#     pred = model([img.to(device)])
#     pred = [{k: v.detach().cpu() for k, v in detection.items()} for detection in pred]
#     predictions = plot_face_mask_scores(img, pred[0])
#     return predictions
#      # Load the image

def get_image(file):
    img = Image.open(file)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    if img.shape[0] == 4:
        rgb = img[:3, :, :] * img[3, :, :]
        img_tensor = torch.stack([rgb[0], rgb[1], rgb[2]], dim=0)
    elif img.shape[0] == 3:
        img_tensor = img
    
    return img_tensor


def predict_file(img_tensor):
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load('model_fasterrcnn_resnet50_fpn.pt', map_location=torch.device('cpu')))
    model.eval()

    device = torch.device("cpu")
    preds = model([img_tensor.to(device)])
    preds = [{k: v.detach().cpu() for k, v in detection.items()} for detection in preds]
    pred = preds[0]
    fig,ax = plt.subplots(1)

    ax.imshow(img_tensor.permute(1, 2, 0))
    
    for box, label,scores in zip(pred["boxes"], pred["labels"],pred["scores"]):
        xmin, ymin, xmax, ymax = box
        score = round(scores.item(),2)
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')       
        ax.add_patch(rect)
        ax.axis('off')

        if label == 1:
            ax.text(xmin, ymin, "Mask:"+str(score), color='g')
        elif label == 2:
            ax.text(xmax, ymax, "Incorrect Mask:"+str(score), color='y')
        else:
            ax.text(xmin, ymin, "No Mask:"+str(score), color='r')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    return buffer

# img = Image.open('D:\Desktop\Client/test.jpg')
# img2 = process_image(img)
# img2.savefig('D:\Desktop\Client/example.jpg')




app = Flask(__name__)

@app.route('/')
def home():
    # return "<h1>Welcome to Mask detection </h1>"
    return render_template('home.html')
   

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = get_image(file)
    response = Response(predict_file(img), mimetype='image/png')
    return response   



if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")




        


