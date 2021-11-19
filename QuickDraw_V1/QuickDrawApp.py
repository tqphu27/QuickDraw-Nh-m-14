from imutils.video import VideoStream
import imutils
import time
import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import os
from PIL import ImageFont, ImageDraw, Image

model = load_model('models/QuickDraw.h5')

def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def get_QD_emojis():
    emojis_folder = 'qd_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder + str(emoji) + '.png', -1))
    return emojis


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y + h, x:x + w] = blend_transparent(image[y:y + h, x:x + w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):

    overlay_img = overlay_t_img[:, :, :3]  
    overlay_mask = overlay_t_img[:, :, 3:]  

    background_mask = 255 - overlay_mask


    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)


    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

  
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))



OPENCV_OBJECT_TRACKERS = {
    	"csrt": cv2.legacy.TrackerCSRT_create,
    	"kcf": cv2.legacy.TrackerKCF_create,
    	"boosting": cv2.legacy.TrackerBoosting_create,
    	"mil": cv2.legacy.TrackerMIL_create,
    	"tld": cv2.legacy.TrackerTLD_create,
    	"medianflow": cv2.legacy.TrackerMedianFlow_create,
    	"mosse": cv2.legacy.TrackerMOSSE_create
	}
tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
initBB = None
vs = cv2.VideoCapture(0)
drawed = []
result = ""
delay_text = 0
emojis = get_QD_emojis()
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
	_, frame = vs.read()
	if frame is None:
		break

	frame = imutils.resize(frame, width=500)

	frame = cv2.flip(frame, 1)
	draw_frame = frame.copy()

	for i in range(1, len(drawed)):
		cv2.line(draw_frame, (drawed[i-1][0], drawed[i-1][1]), (drawed[i][0], drawed[i][1]), (0, 255, 0), 3)
		#cv2.line(blackboard, drawed[i - 1], drawed[i], (255, 255, 255), 7)
	H, W, _ = frame.shape
	if initBB is not None:
		(success, box) = tracker.update(frame)
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(draw_frame, (x, y), (x + w, y + h),
				(255, 0, 0), 2)
			cv2.circle(draw_frame, (x + w//2,y + h//2), 1, (255, 0, 0), -1)
			drawed.append([x + w//2, y + h//2])

	if delay_text > 0:
		cv2.putText(draw_frame, result, (W//2 - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	cv2.imshow("Frame", draw_frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("d"):	
		delay_text = 20

		xmin, ymin = np.min(drawed, 0)
		xmax, ymax = np.max(drawed, 0)

		table_width = xmax - xmin + 20
		table_height = ymax - ymin + 20


		table = np.zeros((table_height, table_width, 3))
		for i in range(1, len(drawed)):
			cv2.line(table, (drawed[i-1][0]-xmin+10, drawed[i-1][1]-ymin+10), (drawed[i][0]-xmin+10, drawed[i][1]-ymin+10), (255, 255, 255), 3)
		table = cv2.resize(table, (28, 28))
		table = table[:, :, 0]/255.
		table = np.expand_dims(table, 0)
		table = np.expand_dims(table, -1)
		table1 = keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
		if True:		
			pred_probab, pred_class = keras_predict(model, table1)
			print(pred_class, pred_probab)
		
			
		drawed = []
		frame = overlay(frame, emojis[pred_class], 400, 250, 100, 100)
  		
 
	elif key == ord("s"):
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		tracker.init(frame, initBB)
	elif key == ord("q"):
		break
  
	delay_text -= 1
 
cv2.imshow("Frame", frame)

vs.release()

cv2.destroyAllWindows()