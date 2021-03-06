import cv2 as cv
import numpy as np

def draw_boxes(img,boxes,classesIds,class_labels,confidences,idex):
    if idex is not None:
        for i in idex.flatten():
            x,y,w,h=boxes[i].astype("int")
            label=class_labels[classesIds[i]]
            cv.rectangle(img,(x-w/2,y-w/2),(x-w/2,y-w/2),(0,255,0),3)
            cv.putText(img,str(label),(x-w/2,y-w/2),1,(0,255,0),5,cv.LINE_AA)
    return img
            
def out_transformation(out,width, height):
    boxes=[]
    confidences=[]
    classesIds=[]
    for i in out:
        for k in i:
            
            scores=k[5:]
            classes=np.argmax(scores)
            confidence=scores[classes]
            if confidence>0.5:
                confidences.append(float(confidence))
                box=k[1:5]* np.array([width,height,width,height],dtype=float)
                
                boxes.append(box)                    
                classesIds.append(classes)
    return boxes,confidences,classesIds

def infer_image(net,layer_names,img,class_labels,width,height,iou_thresh):
    blob = cv.dnn.blobFromImage(img,1/255,(416,416),swapRB=True)
    net.setInput(blob)
    
    out=net.forward(layer_names)
    
    boxes,confidences,classesIds=out_transformation(out,width, height)
    
    idex=cv.dnn.NMSBoxes(boxes,confidences,0.5,iou_thresh)
    
    img=draw_boxes(img,boxes,classesIds,class_labels,confidences,idex)
    return img

cam=cv.VideoCapture(0)
fourcc=cv.VideoWriter_fourcc(*"MJPG")
width=int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height=int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
writter=cv.VideoWriter('output.avi',fourcc,30,(width,height),True)

weights=r'C:\ML\pytorch\YOLO_pytorch\Detectx-Yolo-V3\yolov3.weights'
config=r'C:\ML\pytorch\YOLO_pytorch\Detectx-Yolo-V3\cfg\yolov3.cfg'
class_labels=r'C:\ML\pytorch\YOLO_pytorch\Detectx-Yolo-V3\data\coco.names'

iou_thresh=0.4

with open(class_labels, 'r') as f:
    class_labels= [line.strip() for line in f.readlines()]
    
net=cv.dnn.readNet(weights, config)

layer_names = net.getLayerNames()
layer_names=[layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
while cam.isOpened():
    _,frame=cam.read()
    frame=infer_image(net,layer_names,frame,class_labels,width,height,iou_thresh)
    writter.write(frame)
    cv.imshow('output',frame)
    if cv.waitKey(10) & 0xFF==27:
        break
cam.release()
writter.release()
cv.destroyAllWindows()    
            
            
