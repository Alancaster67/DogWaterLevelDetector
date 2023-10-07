import cv2
from picamera2 import Picamera2
import time
import numpy as np
from PIL import Image
from twilio.rest import Client
from tflite_runtime.interpreter import Interpreter
from tflite_support.task import vision
from tflite_support.task import core

model_path = 'model_sigmoid_data_aug.tflite'
interpreter = Interpreter(model_path=model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
#interpreter = tflite.Interpreter(model_path='model.tflite')
#interpreter.allocate_tensors()
#classify_lite = interpreter.get_signature_runner('serving_default')
#output = interpreter.get_output_details()[0]  # Model has single output.
#input = interpreter.get_input_details()[0]  # Model has single input.
#print(interpreter.get_signature_list())

#base_options = core.BaseOptions(file_name='model.tflite')
#options = vision.ImageClassifierOptions(base_options=base_options)
#classifier = vision.ImageClassifier.create_from_options(options)

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

#cv2.imwrite(f"/home/alancaster67/Documents/DogWaterLevelDetector/Images/HasWater/{time.strftime('%Y%m%d%H%M%S',time.gmtime())}.jpg", bowl_img)
rgb_image = cv2.cvtColor(bowl_img, cv2.COLOR_BGR2RGB)

#img = np.float32(np.array(rgb_image))
img = np.array(rgb_image)
processed_image = np.expand_dims(img, axis=0)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], processed_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]
#tensor_image = vision.TensorImage.create_from_array(rgb_image)
#interpreter.set_tensor(input['index'], bowl_img)
#interpreter.invoke()
#predictions_lite = classify_lite(rescaling_input=rgb_image)['outputs']
#classifier.classify(tensor_image)
LABELS = ['HasWater','NeedsWater']
print(predictions)
print(LABELS[np.argmax(predictions)])