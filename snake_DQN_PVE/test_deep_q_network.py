#!/usr/bin/env python
from __future__ import print_function
from sdata import SNK
import tensorflow as tf
import random
import numpy as np
import sdraw as dr
import time
import traceback
import cv2

GAME = 'snake' # the name of the game being played for log files
ACTIONS = 8 # number of valid actions
GAMMA = 0.99 # decay rate of past observations

TEST_EPSILON = 0
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
MOVE_DICT = {0:(1,0), 1:(1,1), 2:(0,1), 3:(-1,1), 4:(-1,0), 5:(-1,-1), 6:(0,-1), 7:(1,-1)}
STRIDE1 = 2
STRIDE2 = 2
STRIDE3 = 1
PICTURESIZE = 80

TIME_SWITCH = True
DRAW_PICTURE = False


def record_time():
    if TIME_SWITCH is True:
        return time.clock()
    else:
        return None



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def createNetwork():
    # network weights
    W_conv1 = weight_variable([3, 3, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, PICTURESIZE, PICTURESIZE, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, STRIDE1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, STRIDE2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, STRIDE3) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    # fully connected layer
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def testNetwork(s, readout, h_fc1, sess):

    # get state picture
    mysnk = SNK()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    all_x_t, x_t, r_0 = dr.draw_map(mysnk)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 80*80*4

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training

    t = 0
    x_t1_0 = None
    EnergyJson = mysnk.GetCurrentEnergy()
    if EnergyJson['result']['status'] == 0:
        rt_beforAction = EnergyJson['result']['Energy']
    else:
        print("GetEnergy failed\n")
        rt_beforAction = 39

    while "crazy snake" != "dead snake":
        # choose an action epsilon greedily
        loop_time0 = record_time()
        readout_org = readout.eval(feed_dict={s : [s_t]})
        readout_t = readout_org[0]
        # hidden1 = h_conv1.eval(feed_dict={s : [s_t]})
        '''
        for i in range(32):
            cv2.imshow(str(i), hidden1[0][:][:][i])
        '''

        a_t = np.zeros([ACTIONS])

        if  random.random() <= TEST_EPSILON:
            print("------Random  Action------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        try:

            if sum(a_t) != 1:
                raise ValueError('Multiple input actions!')
            # if a_t[0] != 1:
            move_time0 = record_time()
            mysnk.Move(MOVE_DICT[action_index][0], MOVE_DICT[action_index][1])
            move_time1 = record_time()
            if TIME_SWITCH:
                print("Move use time:\t", move_time1 - move_time0)

            # get_isDead_Time0 = record_time()
            # TerminalJson = mysnk.GetIsDead()
            # if TerminalJson['result']['status'] == 0:
            #     terminal = TerminalJson['result']['IsDead']
            # else:
            #     print ("Get IsDead failed!\n")
            #     continue
            # get_isDead_Time1 = record_time()
            # if TIME_SWITCH:
            #     print("Get is Dead us time:\t", get_isDead_Time1-get_isDead_Time0)
            #
            # if terminal:
            #     print("Player dead!")
            #     r_t = -12
            #     time.sleep(5)  # wait for restart button to shown on screen
            #     mysnk.Restart()
            #     time.sleep(2) # make sure next loop will obtain correct picture & energy
            #
            # else:
            #     picture_time0 = record_time()
            #     all_x_t1, x_t1, rt_afteraction = dr.draw_map(mysnk)
            #     if DRAW_PICTURE:
            #         cv2.imshow("src", all_x_t1)
            #         cv2.waitKey(1)
            #
            #     picture_time1 = record_time()
            #     if TIME_SWITCH:
            #         print("Get Picture and status use time:\t", picture_time1 - picture_time0)
            #     r_t = max((rt_afteraction - rt_beforAction) / 5, 0.0002)


            picture_time0 = record_time()
            all_x_t1, x_t1, rt_afteraction = dr.draw_map(mysnk)
            if x_t1 is None:  # get picture failed
                TerminalJson = mysnk.GetIsDead()
                if TerminalJson['result']['status'] == 0:
                    terminal = TerminalJson['result']['IsDead']
                    if terminal:  # definately dead
                        r_t = -12
                        rt_afteraction = rt_beforAction
                        x_t1 = x_t1_0
                        print("Player dead!")
                        time.sleep(5)
                        mysnk.Restart()
                        time.sleep(2)

                    else:
                        print("Picture get failed!\n")
                        continue
                else:
                    print("Get IsDead failed!\n")
                    continue
            else:
                r_t = max((rt_afteraction - rt_beforAction) / 5, 0.0002)
                if DRAW_PICTURE:
                    cv2.imshow("src", all_x_t1)
                    cv2.waitKey(1)
            picture_time1 = record_time()
            if TIME_SWITCH:
                print("Get Picture and status use time:\t", picture_time1 - picture_time0)


        except :
            traceback.print_exc()
            print ("Unexcepted error happend")
            break
        # if x_t1 is None:
        #     print ("Picture get failed \n")
        #     continue

        x_t1 = np.reshape(x_t1, (PICTURESIZE, PICTURESIZE, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # update the old values
        rt_beforAction = rt_afteraction
        x_t1_0 = x_t1
        s_t = s_t1
        t += 1

        print("TIMESTEP", t, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
        if not TIME_SWITCH:
            print ("\n")

        loop_time1 = record_time()
        if TIME_SWITCH:
            print("One loop use time:\t", loop_time1-loop_time0, "\n")


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    testNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
