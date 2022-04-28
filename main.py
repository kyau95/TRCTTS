from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pytesseract # text recognition lib

"""
Currently using a pretrained model to evaluate
And then draw the ROI bounding boxes which 
then will be used with PyTesseract to extract the text
"""

# argument dict for ease of updates
args = {
    "image": "./images/img_2.jpg",
    "east": "./east_text_detection.pb",
    "min_confidence": 0.5,
    "width": 320,
    "height": 320
}

image = cv2.imread(args["image"])
orig = image.copy()
(height, width) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newWidth, newHeight) = (args["width"], args["height"])
ratioWidth, ratioHeight = width / float(newWidth), height / float(newHeight)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newWidth, newHeight))
(height, width) = image.shape[:2]

# two output layers
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"
 ]

# loading the pre-trained EAST text detector
net = cv2.dnn.readNet(args["east"])
# construct a blob from the image and then perform a forward pass
blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []
# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]
 
 	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < args["min_confidence"]:
			continue
		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		# extract rotation angle for the prediction then compute the sin and cos
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
		# use geometry volume to compute  width, height of bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]
		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)
  
		# add bounding box coordinates and probability score to our lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])
  
# apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	startX = int(startX * ratioWidth)
	startY = int(startY * ratioHeight)
	endX = int(endX * ratioWidth)
	endY = int(endY * ratioHeight)
 
	# draw the bounding box on the image
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
 
# display  output image
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)