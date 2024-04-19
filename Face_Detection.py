# face_detect.py
import face_recognition
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from util import set_background
import dlib




# Face recognition - face locations

def detect_faces(image):
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    # Return the face locations
    return face_locations

# YOLO V3

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    
    face_locs = []
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3] 
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        face_locs.append([left,top,right,bottom])
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        # draw_predict(frame, confidences[i], left, top, right, bottom)
    return final_boxes , face_locs

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom

def detect_faces_yolov3(image):
    # Load YOLOv3 model
    net = cv2.dnn.readNetFromDarknet("yolov3-face.cfg", "yolov3-wider_16000.weights\yolov3-wider_16000.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))
    # Remove the bounding boxes with low confidence
    faces, face_locs = post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    return face_locs



# DLIB

def detect_faces_dlib(image):

    face_detector = dlib.get_frontal_face_detector()

    # Find all face locations in the image
    face_locations = face_detector(image, 1)

    # Return the face locations
    return face_locations


def detect_faces_in_image(uploaded_image):
    try:
        # Convert uploaded image to OpenCV format
        img = Image.open(uploaded_image)
        img_cv0 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.cvtColor(np.array(img_cv0), cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        face_locations = detect_faces(img_cv)

        # Draw rectangles around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(img_cv, (left, top), (right, bottom), (0, 255, 0), 3)

        # Display the image with detected faces
        st.image(img_cv, caption='Image with Detected Faces', use_column_width=True)
    except Exception as e:
        st.error("Error detecting faces: {}".format(str(e)))


def detect_faces_in_image_2(uploaded_image):
    try:
        # Convert uploaded image to OpenCV format
        img = Image.open(uploaded_image)
        img_cv0 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        

        # Detect faces in the image
        face_locations = detect_faces_yolov3(img_cv0)

        # Draw rectangles around detected faces
        for (right, bottom, left,top) in face_locations:
            cv2.rectangle(img_cv0, (left, top), (right, bottom), COLOR_YELLOW, 2)
        
        img_cv = cv2.cvtColor(np.array(img_cv0), cv2.COLOR_BGR2RGB)

        # Display the image with detected faces
        st.image(img_cv, caption='Image with Detected Faces', use_column_width=True)
    except Exception as e:
        st.error("Error detecting faces: {}".format(str(e)))

def detect_faces_in_image_3(uploaded_image):
    try:
        # Convert uploaded image to OpenCV format
        img = Image.open(uploaded_image)
        img_cv0 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.cvtColor(np.array(img_cv0), cv2.COLOR_BGR2RGB)
        

        # Detect faces in the image
        face_locations = detect_faces_dlib(img_cv)

        # Draw rectangles around detected faces
        for face in face_locations:
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(img_cv, (l, t), (r, b), (0,255,255), 2)
        

        # Display the image with detected faces
        st.image(img_cv, caption='Image with Detected Faces', use_column_width=True)
    except Exception as e:
        st.error("Error detecting faces: {}".format(str(e)))


