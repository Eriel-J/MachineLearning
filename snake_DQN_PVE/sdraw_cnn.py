from sdata_cnn import SNK

import time
import traceback
import cv2 as cv
import numpy as np

MAP_SIZE = 10040
PVP_MAP_SIZE = 6600
MAP_RADIUS = 4900
PVP_MAP_RADIUS = 3250
ROI_SIZE = 80

BORDER_COLOR = 0
FOOD_COLOR = 255
CORPS_COLOR = 255

PLAYER_HEAD = 110
PLAYER_BODY = 100
ENEMY_HEAD = 20
ENEMY_BODY = 0

SNAKECNT = 'snakeCnt'
MOVEX = 0
MOVEY = 1
LENGTH = 2
ISMAIN = 3
ISACC = 4
ID = 5
WIDTH = 6
ENERGY = 7
DEAD = 8
HEADX = 9
HEADY = 10
KILLCNT = 11
BODYCNT = 12

FOODID = 0
POSX = 1
POSY = 2
RADIUS = 3
FOODENERGY = 4
TYPE = 5

def EnemyAround(matrix):
    if np.any(matrix[:,] == ENEMY_BODY):
        Enemy = True
    elif np.any(matrix[:,] == ENEMY_HEAD):
        Enemy = True
    else:
        Enemy = False
    return  Enemy

def CorpsAround(npArray):
    Corps = False
    if np.any(npArray[:,] == CORPS_COLOR):
        Corps = True
    return Corps



def draw_map(snkinfo, MAP_RATIO=10):
    try:
        total_info = snkinfo.GetAll(MAP_RATIO * ROI_SIZE)
        total_info = eval(total_info)
        if total_info['tp'] != 1:
            return None, None, -1.0, -1, None
        data = total_info['AllSnakeInfo']
        # print (data)
        map_resize = (int(round(MAP_SIZE / MAP_RATIO)), int(round(MAP_SIZE / MAP_RATIO)))
        map = np.zeros(map_resize, np.uint8)
        map += 120
        snake_num = data["snakeCnt"]
        main_index = -1
        for main_index in range(snake_num):
            if data[main_index][0][ISMAIN]:
                # print(data[main_index][0])
                break
            else:
                continue
        if main_index == -1:
            print("Sdraw: There's no snake! player dead!")
            return [], None, -1.0, -1, True
        if data[main_index][0][DEAD]:
            print("sdraw: player dead!")
            return [], None, -1.0, -1, True
        if not data[main_index][0][ISMAIN]:
            print("Cannot find main snake!")
            return [], None, -1.0, None, None
        player_energy = data[main_index][0][ENERGY]
        kill_cnt = data[main_index][0][KILLCNT]


        for i in range(snake_num):
            if i == main_index:
                snake_head = PLAYER_HEAD
                snake_body = PLAYER_BODY
            else:

                snake_head = ENEMY_HEAD
                snake_body = ENEMY_BODY

            snake_width = int(round(data[i][0][WIDTH] / MAP_RATIO))

            body = data[i][1]

            for j in range(len(body)):
                body[j][1] = MAP_SIZE - body[j][1]

            for j in range(len(body) - 1):
                start_point = (int(round(body[j][0] / MAP_RATIO)), int(round(body[j][1] / MAP_RATIO)))
                end_point = (int(round(body[j + 1][0] / MAP_RATIO)), int(round(body[j + 1][1] / MAP_RATIO)))
                cv.line(map, start_point, end_point, snake_body, snake_width)

            head_point = (
            int(round(data[i][0][HEADX] / MAP_RATIO)), int(round((MAP_SIZE - data[i][0][HEADY]) / MAP_RATIO)))
            cv.circle(map, head_point, int(snake_width / 2), snake_head, thickness=-1)
            if i == main_index:
                roi_center = head_point

            center_point = (int(round(MAP_SIZE / MAP_RATIO / 2)), int(round(MAP_SIZE / MAP_RATIO / 2)))
            radius = int(round(MAP_RADIUS / MAP_RATIO))
            cv.circle(map, center_point, radius, BORDER_COLOR, thickness=int(round(MAP_RATIO / 2)))

        try:
            foods = total_info['foods']
            # print(foods)
            for ii in foods.keys():
                if ii == 'foodCnt':
                    continue
                item = foods[ii]
                pos = (int(round(item[POSX] / MAP_RATIO)), int(round((MAP_SIZE - item[POSY]) / MAP_RATIO)))
                radius = int(round(item[RADIUS] / MAP_RATIO))
                cv.circle(map, pos, radius + 1, FOOD_COLOR, thickness=-1)

        except:
            traceback.print_exc()
            print("ZERO-foods: ", foods)
            return None, None, -1.0, -1, False

        roi_pic = cv.getRectSubPix(map, (ROI_SIZE, ROI_SIZE), roi_center)
        # roi = roi_pic.astype(float)
        # roi = roi/120.0-1.0
        # roi = roi_pic -240
        return roi_pic, roi_pic, player_energy, kill_cnt, False
    except:
        traceback.print_exc()
        print("totalInfo: ", total_info)

    return None, None, -1.0, False


if __name__ == '__main__':
    start = time.clock()
    mysnk = SNK()
    map, energy, is_dead = draw_map(mysnk)
    print(is_dead)
    # cv.imshow("map", map)
    end = time.clock()
    cv.imshow("roi", map)

    print("%f" % (end - start))
    cv.waitKey(0)