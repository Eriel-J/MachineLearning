#!/usr/bin/env python
from __future__ import print_function
from sdata import SNK
import tensorflow as tf
import random
import numpy as np
from collections import deque
import sdraw as dr
import time
import traceback
import cv2
import math


GAME = 'snake' # the name of the game being played for log files
ACTIONS = 8 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 400000. # frames over which to anneal epsilon
INITIAL_EPSILON = 0.7 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
TEST_EPSILON = 0.1
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
MOVE_DICT = {0:(1,0), 1:(1,1), 2:(0,1), 3:(-1,1), 4:(-1,0), 5:(-1,-1), 6:(0,-1), 7:(1,-1)}
STRIDE1 = 2
STRIDE2 = 2
STRIDE3 = 1
PICTURESIZE = 80
TRAIN_MODE = True
CONTINUE_TRAIN = True
TIME_SWITCH = True
DRAW_PICTURE = False
TENSOR_BOARD = True
INITIAL_DEAD_REPEAT = 10
FINAL_DEAD_REPEAT = 15
LAST_T = 0
LAST_EPSILON = 0.000111

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

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(5 * 1e-7).minimize(cost)
    # (self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,use_locking=False, name="Adam")

    # get state picture
    mysnk = SNK()
    #allstate, game_state = dr.draw_map(mysnk)

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_"  + "/readout.txt", 'w')
    h_file = open("logs_"  + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    all_x_t, x_t, rt_beforAction,kill_cnt_before = dr.draw_map(mysnk)
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
    epsilon = LAST_EPSILON if CONTINUE_TRAIN else INITIAL_EPSILON
    x_t1 = None
    if CONTINUE_TRAIN:
        t = LAST_T + 1
    else:
        t = 0
    dead_repeat = INITIAL_DEAD_REPEAT

    if TENSOR_BOARD:
        summary_writer = tf.summary.FileWriter("tensorboard", sess.graph)
        train_cost = bias_variable([1])
        tf.summary.scalar('traincost', tf.reduce_sum(train_cost))
        merged_summary_op = tf.summary.merge_all()


    while "crazy snake" != "dead snake":
        loop_time0 = record_time()
        readout_org = readout.eval(feed_dict={s : [s_t]})
        readout_t = readout_org[0]
        # hidden1 = h_conv1.eval(feed_dict={s : [s_t]})
        '''
        for i in range(32):
            cv2.imshow(str(i), hidden1[0][:][:][i])
        '''

        a_t = np.zeros([ACTIONS])
        if TRAIN_MODE:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                print("------Trained  Action------")
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

        else: # test_mode
            if  random.random() <= TEST_EPSILON:
                print("------Random  Action------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1


        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # scale up dead_repeat
        # if t < 100000:
        #     dead_repeat += round((FINAL_DEAD_REPEAT - INITIAL_DEAD_REPEAT) / 100000)

        # run the selected action and observe next state and reward
        try:

            if sum(a_t) != 1:
                raise ValueError('Multiple input actions!')
            # if a_t[0] != 1:
            move_time0 = record_time()
            mysnk.Move(MOVE_DICT[action_index][0], MOVE_DICT[action_index][1])
            move_time1 = record_time()
            if TIME_SWITCH:
                print("Move use time:\t", move_time1 - move_time0)

            get_isDead_Time0 = record_time()
            TerminalJson = mysnk.GetIsDead()
            if TerminalJson['result']['status'] == 0:
                terminal = TerminalJson['result']['IsDead']
            else:
                print ("Get IsDead failed!\n")
                continue
            get_isDead_Time1 = record_time()
            if TIME_SWITCH:
                print("Get is Dead us time:\t", get_isDead_Time1-get_isDead_Time0)

            if terminal:
                print("Player dead!")
                r_t = -12
                kill_cnt_after = 0
                time.sleep(5)  # wait for restart button to shown on screen
                mysnk.Restart()
                time.sleep(2) # make sure next loop will obtain correct picture & energy

            else:
                picture_time0 = record_time()
                all_x_t1, x_t1, rt_afteraction, kill_cnt_after = dr.draw_map(mysnk)
                if DRAW_PICTURE:
                    cv2.imshow("src", all_x_t1)
                    cv2.waitKey(1)

                picture_time1 = record_time()
                if TIME_SWITCH:
                    print("Get Picture and status use time:\t", picture_time1 - picture_time0)
                r_t = min((rt_afteraction - rt_beforAction)/5.0,11)
                r_t = max(r_t, 0.0002)
                r_t += (kill_cnt_after - kill_cnt_before)*12


        except :
            traceback.print_exc()
            print ("Unexcepted error happend,epsilon = ", epsilon)
            break

        if x_t1 is None:
            print ("Picture get failed \n")
            continue
        x_t1 = np.reshape(x_t1, (PICTURESIZE, PICTURESIZE, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        if r_t == 0.0002:
            if random.random() < 0.5:
                D.append((s_t, a_t, r_t, s_t1, terminal))
        else:
            D.append((s_t, a_t, r_t, s_t1, terminal))
        # if r_t > 0.8 : # increase samples of eating large energy such as corpse
        #     times = round(10*(math.log(3 + r_t)))
        #     D.extend([(s_t, a_t, r_t, s_t1, terminal)] * times)
        # if r_t < 0: # increase samples of get killed
        #     D.extend([(s_t, a_t, r_t, s_t1, terminal)] * dead_repeat)

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if train mode or done observing
        if TRAIN_MODE and ((t-LAST_T if CONTINUE_TRAIN else t) > OBSERVE):
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
            cost_value = cost.eval(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})
            print ("  / COST_absolute: ", cost_value, " / COST_normalize", cost_value/np.max(readout_t))

            # tensorboard
            if TENSOR_BOARD:
                summary_str = sess.run(merged_summary_op, feed_dict={train_cost: [cost_value]})
                summary_writer.add_summary(summary_str, t)


        # update the old values
        rt_beforAction = rt_afteraction
        kill_cnt_before = kill_cnt_after
        s_t = s_t1
        t += 1


        # save progress every 10000 iterations
        if TRAIN_MODE and (t % 10000 == 0):
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE :
            state = "observe"
        elif OBSERVE < t <=  + OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        if not TIME_SWITCH:
            print ("\n")

        loop_time1 = record_time()
        if TIME_SWITCH:
            print("One loop use time:\t", loop_time1-loop_time0, "\n")

        # print
        # write info to files
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)


def getEnergy(snk):
    JsonObj = snk.GetCurrentEnergy()
    if JsonObj['result']['status'] == 0:
        return JsonObj['result']['Energy']
    else:
        print("GetEnergy failed\n")
        return None


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
