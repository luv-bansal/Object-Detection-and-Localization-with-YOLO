import cv2 as cv
import numpy as np

def draw_boxes(img,boxes,classesIds,class_labels,confidences,idex):
    if idex is not None:
        for i in idex.flatten():
            x,y,w,h=boxes[i]
            label=class_labels[classesIds[i]]
            cv.rectangle(img,(x-w/2,y-w/2),(x-w/2,y-w/2),(0,255,0),3)
            cv.putText(img,str(label),(x-w/2,y-w/2),1,(0,255,0),5,cv.LINE_AA)
    return img
            
def out_transformation(out,width, height):
    boxes=[]
    confidences=[]
    classesIds=[]
    for i in out:
        for j in i:
            for k in j:
                scores=k[5:]
                classes=np.argmax(scores)
                confidence=scores[classes]
                if confidence>0.5:
                    confidences.append(confidence)
                    box=k[1:5]* np.array([width,height,width,height])
                    
                    boxes.append(box)                    
                    classesIds.append(classes)
    return boxes,confidences,classesIds
def infer_image(net,layer_names,img,class_labels,width,height,iou_thresh):
    blob = cv.dnn.blobFromImage(img,1/255,(416,416),swapRB=True)
    net.setInput(blob)
    
    out=net.forward()
    
    boxes,confidences,classesIds=out_transformation(out,width, height)
    
    idex=cv.dnn.NMSBoxes(boxes,confidences,0.5,iou_thresh)
    
    img=draw_boxes(img,boxes,classesIds,class_labels,confidences,idex)
    return img

cam=cv.vedioCapture(0)
fourcc=cv.vedioWriter_fourcc(*'XVID')
width=cam.get(cv.CAP_PROP_FRAME_WIDTH)
height=cam.get(cv.CAP_PROP_FRAME_HEIGHT)
writter=cv.vedioWriter('outputavi',fourcc,20,(width,height))
weights='C:\ML\Darknet\darknet\yolov3.weights'
config='C:\ML\Darknet\darknet\cfg\yolov3.cfg'
class_labels='C:\ML\Darknet\darknet\data\coco.names'
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
    

    
    
            
            
