import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
image=cv2.VideoCapture(f"testkn.mp4")
# image_path = 'img.jpg'
# image = cv2.imread(image_path)
# image = cv2.resize(image,(640,480))
with open('coco.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
def getObjects(frame):


    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.numpy()  
        classids = result.boxes.cls.numpy()

   

    colors = np.random.uniform(0, 255, size=(len(class_names), 3))

    for box,classid in zip(boxes,classids):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(1)]
        lab = str(class_names[int(classid)])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, lab, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
    # classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    # #print(classIds,bbox)
    # if len(objects) == 0: objects = classNames
    # objectInfo =[]
    # if len(classIds) != 0:
    #     for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    #         className = classNames[classId - 1]
    #         if className in objects:
    #             objectInfo.append([box,className])
    #             if (draw):
    #                 cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #                 cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
    #                 cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #                 cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
    #                 cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    # return img,objectInfo

while True:
    ret,frame=image.read()
    if ret:
        frame=getObjects(frame)

        cv2.imshow("frame", frame)
        cv2.waitKey(1)
cv2.destroyAllWindows()
