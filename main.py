from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pytesseract # text recognition lib

# argument dict for ease of updates
args = {
    "image": "./images/test/img_3.jpg",
    "east": "./east_text_detection.pb",
    "min_confidence": 0.5,
    "width": 320,
    "height": 320
}

"""
training model to evaluate then draw the ROI bounding boxes 
then will be used with PyTesseract to extract the text
gTTS will then take the extracted text and read it out
"""
image = cv2.imread(args["image"])
orig = image.copy()
(height, width) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(new_width, new_height) = (args["width"], args["height"])
ratio_width, ratio_height = width / float(new_width), height / float(new_height)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (new_width, new_height))
(height, width) = image.shape[:2]

# two output layers
layer_names = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"
 ]

# loading the pre-trained EAST text detector
net = cv2.dnn.readNet(args["east"])
# construct a blob from the image and then perform a forward pass
blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layer_names)

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(rows, cols) = scores.shape[2:4]
rects = []
confidences = []
# loop over the number of rows
for y in range(0, rows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	score_data = scores[0, 0, y]
	x_data0 = geometry[0, 0, y]
	x_data1 = geometry[0, 1, y]
	x_data2 = geometry[0, 2, y]
	x_data3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]
 
 	# loop over the number of columns
	for x in range(0, cols):
		# if our score does not have sufficient probability, ignore it
		if score_data[x] < args["min_confidence"]:
			continue
		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		# extract rotation angle for the prediction then compute the sin and cos
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
		# use geometry volume to compute  width, height of bounding box
		h = x_data0[x] + x_data2[x]
		w = x_data1[x] + x_data3[x]
		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
		end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
		start_x = int(end_x - w)
		start_x = int(end_y - h)
  
		# add bounding box coordinates and probability score to our lists
		rects.append((start_x, start_x, end_x, end_y))
		confidences.append(score_data[x])
  
# apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
# loop over the bounding boxes
for (start_x, start_x, end_x, end_y) in boxes:
	start_x = int(start_x * ratio_width)
	start_x = int(start_x * ratio_height)
	end_x = int(end_x * ratio_width)
	end_y = int(end_y * ratio_height)
 
	# draw the bounding box on the image
	cv2.rectangle(orig, (start_x, start_x), (end_x, end_y), (0, 255, 0), 2)
 
# display  output image
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)