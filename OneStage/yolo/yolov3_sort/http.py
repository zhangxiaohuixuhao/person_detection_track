# -*- coding:utf-8 -*-
import urllib.requests as urllib2
import json
import collections

def RobotLogin(robot_id,robot_name):
    msg = collections.OrderedDict()
    msg = {'sequenceId': robot_id, 'robotName': robot_name}
    return msg

def Post(url,msg):
    print(msg)
    data_json = json.dumps(msg)
    data = 'data='+data_json
    print(data)
    request = urllib2.Request(url, data)
    response = urllib2.urlopen(request)
    return response.read()

if __name__=='__main__':
    url = 'http://180.108.46.7:10808/web/api/robot/login'
    Post(url, RobotLogin('12345678','testrobot'))