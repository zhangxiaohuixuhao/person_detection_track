#encoding: utf-8
# import the necessary packages
global img
global point1, point2
import cv2


PointsChoose = [] #线段两个点的坐标集合
pointsCount = 0 #鼠标左键点击次数
line = [] #经过处理之后的直线段与图像两边之间的交点

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture('rtsp://184.72.239.149/vod/mp4://BigBuckBunny_175k.mov')
grabbed, frame = vs.read()
writer = None
(H, W) = frame.shape[:2]

####鼠标点击选点
def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    global lsPointsChoose, PointsChoose  #存入选择的点
    global pointsCount   #对鼠标按下的点计数
    global img2
    img2 = img.copy()   #此行代码保证每次都重新再原图画  避免画多了

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


#############鼠标左键选择适合的点划线##############
line_point = 0
if len(PointsChoose) == 0:
    while line_point == 0:
        # read the next frame from the file
        (grabbed, img) = vs.read()
        (H, W) = img.shape[:2]
        cv2.imshow('src', img)
        cv2.namedWindow('src')
        cv2.setMouseCallback('src', on_mouse)
        cv2.imshow('src', img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            line_point += 1
            # print(PointsChoose)
            cv2.destroyAllWindows()

line = extension(PointsChoose)
with open("line.txt", 'w') as f:
    f.write(str(line))
print(line)