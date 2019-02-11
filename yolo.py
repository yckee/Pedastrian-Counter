import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from tracker import Tracker

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Put info about trackable objects on frame
def drawObjects(objects):
    for id, centroid in objects.items():
        label = f'ID {id}'
        cv.putText(frame, label, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 255), -1)
      
# Detect objects, remove overlaps and track 
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == 0:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2) 
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    centroids = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        center_x = int(box[0] + box[2]/2)
        center_y = int(box[1] + box[3]/2)
        centroids.append((center_x,center_y))
    
    return tracker.update(centroids)



if __name__ == "__main__":

    #Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to video file.')
    args = parser.parse_args()

    # Initialize the parameters
    confThreshold = 0.5  #Confidence threshold
    nmsThreshold = 0.4   #Non-maximum suppression threshold
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image


    # Load and configure NN
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    tracker = Tracker(20)


    if not os.path.isfile(args.path):
        print("Input video file ", args.path, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.path)

    outputFile = args.path[:-4]+'_out.avi'
    vid_writer = cv.VideoWriter(
        outputFile, 
        cv.VideoWriter_fourcc('X','V','I','D'),
        cap.get(cv.CAP_PROP_FPS),
        (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        )

    #Proccess video frame by frame
    while True:
        
        success, frame = cap.read()
        if not success:
            print("Proccessing complete")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            cap.release()
            break

        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)

        outs = net.forward(getOutputsNames(net))
        objects = postprocess(frame, outs)
        drawObjects(objects)

        label = f'Number of people: {len(objects)}'
        cv.putText(frame, label, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

        vid_writer.write(frame.astype(np.uint8))

        cv.imshow('Counter', cv.resize(frame,(0,0), fx=0.5, fy=0.5))
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break