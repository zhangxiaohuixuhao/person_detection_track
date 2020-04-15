# -*- coding: UTF-8 -*-
# import the necessary packages
import os
from sort import *
import sys
sys.path.append('.')
from detect import *
import requests
import json
import collections
import time
from xml.dom import minidom
import argparse
import ast
import datetime
from skimage.io import imshow
import glob
import threading
import copy
import cv2
tracker = Sort()  # 跟踪算法DeepSORT
memory = {}  # 存储前一帧的跟踪信息
counter = 0  # 总经过人数
in_num = 0  # 进入人数
out_num = 0  # 出去人数
misjudgment_num = []  # 已经计数过的ID
PointsChoose = []

# Constant
CAM_IN_TYPE = "cam_in"
CAM_OUT_TYPE = "cam_out"

# Declared global variable
timeinterval = None  # 统计的时间间隔
vs = None  # 视频句柄
status = None  # 0 表示入口， 1 表示出口
line = None  # 交叉线
directions = None  # 进入的法向量
url = None    # 后台地址
grabbed = None
img = None
# vs = cv2.VideoCapture('./yolo-obj/20191125-173553.mp4')
(W, H) = (1920, 1080)


def choosepoint():
    line_point = 0
    if len(PointsChoose) == 0:
        while line_point == 0:
            # read the next frame from the file
            cv2.imshow('src', img)
            cv2.namedWindow('src')
            cv2.setMouseCallback('src', on_mouse)
            cv2.imshow('src', img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                line_point += 1
                # print(PointsChoose)
                cv2.destroyAllWindows()

####鼠标点击选点
def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    global PointsChoose  # 存入选择的点
    global img2
    pointsCount = 0  # 对鼠标按下的点计数
    img2 = img.copy()   # 此行代码保证每次都重新再原图画  避免画多了

    if event == cv2.EVENT_LBUTTONDOWN:
        #左键点击
        if pointsCount < 2:
            pointsCount = pointsCount+1
            print('pointsCount:', pointsCount)
            point1 = (x, y)
            print(point1)
            #画出点击的点
            cv2.circle(img2, point1, 10, (0,255,0), 5)
            cv2.imshow('src', img2)
            cv2.waitKey(15)
            # cv2.imshow('src', img2)
            # 将选取的点保存到list列表里
            PointsChoose.append((x,y))  #用于画点
            #将鼠标选的点用直线链接起来
            print(len(PointsChoose))
            for i in range(len(PointsChoose) - 1):
                # print('i', i)
                cv2.line(img2, PointsChoose[i], PointsChoose[i + 1], (0, 0, 255), 5)
            cv2.imshow('src', img2)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                print(PointsChoose)

######判断鼠标选取的点与图像两边的交点
def extension(point):
    point0 = cross_point(point, [(0, 0), (W, 0)])
    point1 = cross_point(point, [(W, 0), (W, H)])
    point2 = cross_point(point, [(0, H), (W, H)])
    point3 = cross_point(point, [(0, 0), (0, H)])
    if check(point0):
        line.append(point0)
    if check(point1):
        line.append(point1)
    if check(point2):
        line.append(point2)
    if check(point3):
        line.append(point3)
    if line[0][0] > line[1][0]:
        tmp = line[0]
        line[0] = line[1]
        line[1] = tmp
    return line

def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0][0] # 取四点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        x = x3
        y = k1 * x + b1
    else:
        y = y3
        x = (y3 - b1) / k1
    return (int(x), int(y))

def check(point):
    if point[0] >= 0 and point[1] >= 0:
        if point[0] <= W and point[1] <= H:
            return True

def readline():
    global line
    if os.path.exists(args["txtpath"]):
        with open(args["txtpath"], 'r') as f:
            lines = f.readlines()
            line = lines[int(args["cameraid"])]
            line = ast.literal_eval(line)
            print(line)
    else:
        choosepoint()
        line = extension(PointsChoose)
        with open(args["txtpath"], 'w') as f:
            f.write(str(line))
        print(line)

#判断每一个ID前一帧与后一帧连成的线段是否与所划线相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

#判断相交直线的两端是怎样分布在直线的两边，判断是进入还是出去
def inout(A, B, C, D, Up_to_Down_is_in=True):
    """
    Written by Zhang Hui, revised by Guanghao, added input flag Up_to_Down_is_in
    :param Up_to_Down_is_in: default is True, which denotes movement from up to down is along in direction.
    :return:
    """
    if lineEquation(A, C, D) == True and lineEquation(B, C, D) == False:
        return True if Up_to_Down_is_in else False
    else:
        return False if Up_to_Down_is_in else True

def inout_v2(From, To, C, D, Cur_dir):
    """
    By Guanghao Zhang.
    This function enables in or out direction judgement for multiple line segments
    Cur_dir: 'down' or 'up'
    :return:
    """
    if Cur_dir == 'down':
        return inout(From, To, C, D, Up_to_Down_is_in=True)
    else:
        return inout(From, To, C, D, Up_to_Down_is_in=False)


def lineEquation(A, B, C):
    tmp = B[1] + (A[0] - B[0]) * (C[1] - B[1])/(C[0] - B[0])
    if tmp <= A[1]:
        return True
    else:
        return False
#####http####
def RobotLogin(time, status, cameraid, timeinterval, in_num, out_num):
    msg = collections.OrderedDict()
    msg = {'timeStamp': time, 'status': status, 'cameraId': cameraid, 'timeInterval': timeinterval, 'InPdtCount': in_num, 'OutPdtCount': out_num}
    return msg

def Post(url, msg):
    print(msg)
    data_json = json.dumps(msg)
    data = "data=" + data_json
    data = requests.post(url=url, data=data, headers={'Content-Type':'application/x-www-form-urlencoded'})
    return data.text
def addPlotting(frame):
    """
    Add line segments to frames;
    :param frame: input cv2 style frame
    :return: none
    """
    global line
    for cur_line in line:
        cv2.line(frame, cur_line[0], cur_line[1], (0,0,255),2)

def addNotation(frame, in_num=0, out_num=0, interval=10., rates=None):
    """
    By Guanghao, add in numbers and out to frame
    :param frame:
    :param in_num: integer
    :param out_num: integer
    :return:
    """
    if len(frame.shape) == 3:
        w, h, _ = frame.shape
    else:
        w, h = frame.shape
    if not rates:
        notation = 'In: %d and Out: %d in %0.1f seconds' % (in_num, out_num, interval)
    else:
        notation = 'In: %d and Out: %d in %0.1f seconds. Rates %0.2f' % (in_num, out_num, interval, rates)
    cv2.putText(frame, notation, (int(w/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

#####detect and track##########
def detectandtrack():
    global memory, misjudgment_num, counter, in_num, out_num, line, img, grabbed
    frameIndex = 0
    while True:
        # grabbed, frame = vs.read()  # todo: 是否每一帧都要处理，应该加入帧丢弃机制
        if grabbed == True:
            # if skip >= 1:  # 每隔一帧跳一帧
            #     skip = 0
            #     continue
            # else:
            #     skip += 1
            time_start = time.time()
            frame = copy.deepcopy(img)
            im = nparray_to_image(frame)  # todo: frame[:,:,::-1]会有漏人，但是frame[:,:,:]不会
            r = detecter(net, meta, im)
            boxes = []
            confidences = []

            for detection in r:
                # loop over each of the detections
                confidence = detection[1:][0]

                if confidence > 0.5:
                    box = np.array(detection[0])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
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
                        cv2.rectangle(frame, (x, y), (w, h), color, 3)
                        # (frame, notation, (int(w/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                        cv2.putText(frame, 'ID:%d' % indexIDs[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            cv2.line(frame, p0, p1, (0, 255, 0), 3)

                            for cur_line, cur_direction in zip(line, directions):
                                if intersect(p0, p1, cur_line[0], cur_line[1]):  # todo: 判断与某条线段相交
                                    if indexIDs[i] not in misjudgment_num:
                                        misjudgment_num.append(indexIDs[i])
                                        counter += 1
                                        if inout_v2(p0, p1, cur_line[0], cur_line[1], cur_direction):
                                            in_num += 1
                                        else:
                                            out_num += 1
                                    break  # todo: 最多只会与一个线段相交，否则会导致bug

                        i += 1
            if len(misjudgment_num) >= 5000:
                misjudgment_num = []
            # frameIndex += 1
            # print(frameIndex)
            time_end = time.time()
            print(time_end - time_start)
            addPlotting(frame)
            addNotation(frame, in_num=in_num, out_num=out_num, interval=(time_end - time_start), rates=frameIndex/(time_end - time_start))
            cv2.imwrite("/DATA/zhanghui/Object-Detection-and-Tracking/OneStage/yolo/yzpark_ip/out/frame-{}.png".format(frameIndex), frame)
            frameIndex += 1
            # cv2.imshow('1', frame)
            # cv2.waitKey(10)
            print(frameIndex)
            if time_end - time_start > int(timeinterval):
                if status == 1:
                    temp = in_num
                    in_num = out_num
                    out_num = temp
                Post(url, RobotLogin(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')), str(status),
                                     str(args["cameraid"]), str(timeinterval), str(in_num), str(out_num)))
                time_start = time.time()
                in_num = 0
                out_num = 0
                frameIndex = 0
                # misjudgment_num = []

def xml_parser():
    """
    By Guanghao Zhang.
    This function reads xml configuration file, leaves line.txt file out for simplification.
    Default xml file path is ./yolo-obj/yz.xml
    It returns line, direction, rstp address, camera type, time interval
    :return: line, direction, rstp, cameratype, timeinterval
    """
    dom = minidom.parse(args["xmlpath"])

    interval = dom.getElementsByTagName('T')
    timeinterval = int(interval[0].firstChild.data)

    cameraid = int(args["cameraid"])
    rtsp = None
    line = None
    direction = None
    camera_type = None

    url = dom.getElementsByTagName('Backbone')[0].getAttribute('address')
    for ele in dom.getElementsByTagName('Camera'):
        if cameraid == int(ele.getAttribute('id')):
            rtsp = ele.getElementsByTagName('rtsp')[0].firstChild.data
            line = ast.literal_eval(ele.getElementsByTagName('line')[0].firstChild.data)
            tmp_type = ele.getElementsByTagName('type')[0].firstChild.data
            direction = ast.literal_eval(ele.getElementsByTagName('direction')[0].firstChild.data)
            if tmp_type == CAM_IN_TYPE:
                camera_type = 0
            elif tmp_type == CAM_OUT_TYPE:
                camera_type = 1
            else:
                raise Exception("Camera Type should be cam_in or cam_out. Error happens for camera %d. in xml file." % cameraid)
            break
    assert camera_type is not None
    return line, direction, rtsp, camera_type, timeinterval, url

def ImageGet():
    global grabbed, img
    # vs = cv2.VideoCapture(rtsp)
    vs = cv2.VideoCapture('/DATA/zhanghui/Object-Detection-and-Tracking/OneStage/yolo/yzpark_ip/yolo-obj/20191125-173553.mp4')
    while vs.isOpened():
        grabbed, img = vs.read()
        # img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('1', img)
        cv2.waitKey(35)
        # print(grabbed)

def main():
    # global grabbed, img
    #readline()
    global timeinterval, vs, status, line, grabbed, img, url, directions
    line, directions, rtsp, status, timeinterval, url = xml_parser()

    tCamera = threading.Thread(target=ImageGet)
    tCamera.start()

    tCamera = threading.Thread(target=detectandtrack())
    tCamera.start()

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cameraid",
                    help="input cameraid", default="0")
    ap.add_argument("-x", "--xmlpath", help="path to xmlfile", default="./yolo-obj/yz.xml")
    args = vars(ap.parse_args())

    # global timeinterval, vs, status, line, grabbed, img, url, directions
    line, directions, rtsp, status, timeinterval, url = xml_parser()

    tCamera = threading.Thread(target=ImageGet)
    tCamera.start()

    tCamera = threading.Thread(target=detectandtrack)
    tCamera.start()
    # main()

