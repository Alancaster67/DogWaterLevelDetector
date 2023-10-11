import cv2
from picamera2 import Picamera2
import time
import numpy as np
from twilio.rest import Client

piCam = Picamera2()
piCam.preview_configuration.main.size=(1280,720)
piCam.preview_configuration.main.format="RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()
time.sleep(2)
PILimg = piCam.capture_image("main")

image = cv2.cvtColor(np.array(PILimg), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

if np.median(gray) < 30:
    #image is not bright enough
    exit()

#gray = cv2.medianBlur(gray,10)
circles = cv2.HoughCircles(gray,
                           cv2.HOUGH_GRADIENT_ALT,
                           dp = 1,
                           minDist = 120,
                           param1 = 300,
                           param2 =0.9,
                           minRadius=100,
                           maxRadius=175
                           )
if circles is None:
#    account_sid = "ACc658503e6f5e0efcec4ac334f0f16b69"
#    auth_token = "d4703fb3df6d7fdcdcc78015ad319d77"
#
#    client = Client(account_sid, auth_token)
#
#    message = client.messages.create(
#        body = "No dog bowl detected under PiCam. Check dog bowl",
#        from_ = "+18665184103",
#        to = "+15206684049"
#        )
    exit()
    
circles = np.uint16(np.around(circles))[0]
#Assume the largest circle is the dog bowl
bowl = circles[(np.argmax(circles[:,-1])),:]

bowl_radius = bowl[2]
bowl_center_row = bowl[1]
bowl_center_col = bowl[0]

#need to try-catch slicing exceptions
bowl_img = image[bowl_center_row - bowl_radius: bowl_center_row + bowl_radius,
                 bowl_center_col - bowl_radius:bowl_center_col + bowl_radius]

bowl_img = cv2.resize(bowl_img, (258,258))

cv2.imwrite(f"/home/alancaster67/Documents/DogWaterLevelDetector/Images/HasWater/{time.strftime('%Y%m%d%H%M%S',time.gmtime())}.jpg", bowl_img)