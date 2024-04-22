# import the opencv library 
import cv2 
import time
from ultralytics import YOLO
import math

model =YOLO('yolov8m-pose.pt')  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    results=model(frame)
    annotated_frame=results[0].plot()
    keypoints =  results[0].keypoints.xy.cpu().numpy()[0]
    print(keypoints)
    # for idx in range(results.shape[0]):
    #     keypoints=results[0].keypoints.xyn.cpu().numpy()[idx]
    #     # x_min,y_min,x_max,y_max = results[0].boxes.xyxyn
    #     x=results[0].boxes.xyxyn
        
    #     print('********************************')
    #     print(results[0].boxes)
    #     print(x[0])
    #     print('********************************')
    #     NOSE            = keypoints[0]
    #     LEFT_EYE        = keypoints[1]
    #     RIGHT_EYE       = keypoints[2]
    #     LEFT_EAR        = keypoints[3]
    #     RIGHT_EAR       = keypoints[4]
    #     LEFT_SHOULDER   = keypoints[5]
    #     RIGHT_SHOULDER  = keypoints[6]
    #     LEFT_ELBOW      = keypoints[7]
    #     RIGHT_ELBOW     = keypoints[8]
    #     LEFT_WRIST      = keypoints[9]
    #     RIGHT_WRIST     = keypoints[10]
    #     LEFT_HIP        = keypoints[11]
    #     RIGHT_HIP       = keypoints[12]
    #     LEFT_KNEE       = keypoints[13]
    #     RIGHT_KNEE      = keypoints[14]
    #     LEFT_ANKLE      = keypoints[15]
    #     RIGHT_ANKLE     = keypoints[16]
    #     len_factor = math.sqrt(((LEFT_SHOULDER[1] - LEFT_HIP[1])**2 + (LEFT_SHOULDER[0] - LEFT_HIP[0])**2 ))
        # if LEFT_SHOULDER[1] > LEFT_ANKLE[1] - len_factor and LEFT_HIP[1] > LEFT_ANKLE[1] - (len_factor / 2) and LEFT_SHOULDER[1] > LEFT_HIP[1] - (len_factor / 2):
            
        #     cv2.rectangle(im0,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(0, 0, 255),
        #         thickness=5,lineType=cv2.LINE_AA)
        #     cv2.putText(im0, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)
        #     bot.sendMessage(receiver_id, "Person Fall Detected")
        #     filename = "D:\\ws\\opencv\\yolo7\\yolov7\\savedImage.jpg"
        #     cv2.imwrite(filename, im0)
        #     bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
        #     os.remove(filename)

    cv2.imshow('frame', annotated_frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 




# example 
