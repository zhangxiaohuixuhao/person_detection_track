#encoding: utf-8
# import the necessary packages
import argparse
import os
import glob
import numpy as np
global img
global point1, point2
import cv2
from ctypes import *
import time
from sort import *
from PIL import Image, ImageDraw, ImageFont

tracker = Sort() #跟踪算法DeepSORT
memory = {} #存储前一帧的跟踪信息
line = [] #经过处理之后的直线段与图像两边之间的交点
counter = 0 #总经过人数
in_num = 0 #进入人数
out_num = 0 #出去人数
misjudgment_num = [] #已经计数过的ID
with open('./line_determine/line.txt', 'r') as f:
    line = eval(f.readline())

vs = cv2.VideoCapture('rtsp://184.72.239.149/vod/mp4://BigBuckBunny_175k.mov')
writer = None
(W, H) = (None, None)

frameIndex = 0

#######darknet模型加载预备##########
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("/DATA/zhanghui/darknet/libdarknet.so", RTLD_GLOBAL) #darknet加载make得到的.so文件
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["./yolo-obj", "yz.names"])
LABELS = open(labelsPath).read().strip().split("\n")

print("[INFO] loading YOLO from disk...")
net = load_net(b"./yolo-obj/yz.cfg",
               b"/DATA/zhanghui/darknet/model/yz_50016.weights", 0)
meta = load_meta(b"./yolo-obj/yz.data")


# Return true if line segments AB and CD intersect判断每一个ID前一帧与后一帧连成的线段是否与所划线相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

#判断相交直线的两端是怎样分布在直线的两边，判断是进入还是出去
def inout(A, B, C, D):
    if lineEquation(A, C, D) == True and lineEquation(B, C, D) == False:
        return True
    else:
        return False

def lineEquation(A, B, C):
    tmp = B[1] + (A[0] - B[0]) * (C[1] - B[1])/(C[0] - B[0])
    if tmp >= A[1]:
        return True
    else:
        return False

#####darknet检测函数
def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    num = c_int(0)
    pnum = pointer(num)
    start = time.time()
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms):
        do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append(((b.x, b.y, b.w, b.h), dets[j].prob[i]))
    free_image(im)
    free_detections(dets, num)
    return res

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    start = time.time()
    grabbed, frame = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    im = nparray_to_image(frame)
    r = detect(net, meta, im)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []

    # loop over each of the layer outputs
    for detection in r:
        # loop over each of the detections
        confidence = detection[1:][0]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > 0.5:
            # scale the bounding box coordinates back relative to
            # the size of the image, keeping in mind that YOLO
            # actually returns the center (x, y)-coordinates of
            # the bounding box followed by the boxes' width and
            # height
            box = np.array(detection[0])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top
            # and and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates,
            # confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    dets = []
    if len(boxes) > 0:
        # loop over the indexes we are keeping
        for i in range(len(boxes)):
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    if len(dets) == 0:
        misjudgment_num = []
    if len(dets) > 0:
        tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
        for i in range(len(previous.keys())):
            if (list(previous.keys())[i] in misjudgment_num) and (list(previous.keys())[i] not in indexIDs):
                misjudgment_num.remove(list(previous.keys())[i])
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    cv2.line(frame, p0, p1, color, 3)
                    if intersect(p0, p1, line[0], line[1]):
                        if indexIDs[i] not in misjudgment_num:
                            misjudgment_num.append(indexIDs[i])
                            counter += 1
                            if inout(p0, p1, line[0], line[1]):
                                in_num += 1
                            else:
                                # in_num -= 1
                                out_num += 1


                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                # text = "{}".format(indexIDs[i])
                # print(text)
                # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1
    if len(misjudgment_num) >= 5000:
        misjudgment_num = []

    end = time.time()
    print(end - start)

    # saves image file
    # cv2.imwrite("output/frame-{}.png".format(frameIndex), cv2charimg)

    # increase frame index
    frameIndex += 1


