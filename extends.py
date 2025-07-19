import numpy as np
import math


def toLegal(angle):
    angle = int(angle)
    if int(angle) < 0:
        angle = 0
    if len(str(angle)) == 2:
        angle = str('0') + str(angle)
    elif len(str(angle)) == 1:
        angle = str('00') + str(angle)

    return str(angle)


def cut_angle(int_data):
    int_data = int(int_data)
    int_data = int_data - 20
    if int_data < 0:
        int_data = 0
    return int_data


zb_x = np.zeros([33])
zb_y = np.zeros([33])
zb_z = np.zeros([33])


def angle(zb1, zb2, zb3):
    dx1 = zb_x[zb2] - zb_x[zb1]
    dy1 = zb_y[zb2] - zb_y[zb1]
    dx2 = zb_x[zb2] - zb_x[zb3]
    dy2 = zb_y[zb2] - zb_y[zb3]
    # print(dx1, dx2, dy1, dy2)
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def angle1(zb1, zb2):
    dx1 = zb_x[zb2] - zb_x[zb1]
    dy1 = zb_y[zb2] - zb_y[zb1]
    dx2 = zb_x[zb2] - zb_x[zb2]
    dy2 = zb_y[zb2] - zb_y[zb1]
    # print(dx1, dx2, dy1, dy2)
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle
