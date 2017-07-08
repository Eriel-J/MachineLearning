#!/usr/bin/env python
from __future__ import print_function
from sdata_cnn import SNK
import tensorflow as tf
import random
import numpy as np
from collections import deque
import sdraw_cnn as dr
import time
import traceback
import cv2

GAME = 'snake' # the name of the game being played for log files
ACTIONS = 16 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 300. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
INITIAL_EPSILON = 0.36 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
# MOVE_DICT = {0:(1,0), 1:(1,1), 2:(0,1), 3:(-1,1), 4:(-1,0), 5:(-1,-1), 6:(0,-1), 7:(1,-1)}
MOVE_DICT = {0:(1,0,0),1:(1,1,0),2:(0,1,0),3:(-1,1,0),4:(-1,0,0),5:(-1,-1,0),6:(0,-1,0),7:(1,-1,0),
             8:(1,0,1),9:(1,1,1),10:(0,1,1),11:(-1,1,1),12:(-1,0,1),13:(-1,-1,1),14:(0,-1,1),15:(1,-1,1)
             }
STRIDE = 1
PICTURESIZE = 80
TRAIN_MODE = False
CONTINUE_TRAIN = False
TIME_SWITCH = False
DRAW_PICTURE = False
TENSOR_BOARD = False
INITIAL_DEAD_REPEAT = 10
FINAL_DEAD_REPEAT = 15
LAST_T = 881749
LAST_EPSILON = 0.2
TEST_EPSILON = 0.01


class SnakeDqn(object):

    def __init__(self):
        self.mysnk = SNK()
        self.D = deque()
        self.last_back_time = time.clock()
        self.last_back_time = 0
        self.sess = None
        self.epsilon = None
        self.t = None
        self.train_step = None
        self.dead_repeat = 0
        self.saver = None
        self.summary_writer = None
        self.train_cost = None
        self.merged_summary_op = None

    @staticmethod
    def record_time():
        if TIME_SWITCH is True:
            return time.clock()
        else:
            return None

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    @staticmethod
    def createNetwork():
        # network weights
        W_conv1 = SnakeDqn.weight_variable([3, 3, 4, 32])
        b_conv1 = SnakeDqn.bias_variable([32])

        W_conv1_2 = SnakeDqn.weight_variable([3, 3, 32, 32])
        b_conv1_2 = SnakeDqn.bias_variable([32])

        W_conv1_3 = SnakeDqn.weight_variable([3, 3, 32, 32])
        b_conv1_3 = SnakeDqn.bias_variable([32])

        W_conv2 = SnakeDqn.weight_variable([3, 3, 32, 64])
        b_conv2 = SnakeDqn.bias_variable([64])

        W_conv3 = SnakeDqn.weight_variable([3, 3, 64, 64])
        b_conv3 = SnakeDqn.bias_variable([64])

        W_fc1 = SnakeDqn.weight_variable([6400, 512])
        b_fc1 = SnakeDqn.bias_variable([512])

        # W_fc2 = SnakeDqn.weight_variable([1600, 512])
        # b_fc2 = SnakeDqn.bias_variable([512])

        W_fc3 = SnakeDqn.weight_variable([512, ACTIONS])
        b_fc3 = SnakeDqn.bias_variable([ACTIONS])

        # input layer
        s = tf.placeholder("float", [None, PICTURESIZE, PICTURESIZE, 4])

        # hidden layers
        h_conv1_1 = tf.nn.relu(SnakeDqn.conv2d(s, W_conv1, STRIDE) + b_conv1)
        h_pool1_1= SnakeDqn.max_pool_2x2(h_conv1_1)
        h_conv1_2= tf.nn.relu(SnakeDqn.conv2d(h_pool1_1, W_conv1_2, STRIDE) + b_conv1_2)
        h_pool1_2 = SnakeDqn.max_pool_2x2(h_conv1_2)
        h_conv1_3 = tf.nn.relu(SnakeDqn.conv2d(h_pool1_2, W_conv1_3, STRIDE) + b_conv1_3)
        h_pool1_3 = SnakeDqn.max_pool_2x2(h_conv1_3)

        h_conv2 = tf.nn.relu(SnakeDqn.conv2d(h_pool1_3, W_conv2, STRIDE) + b_conv2)
        # h_pool2 = SnakeDqn.max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(SnakeDqn.conv2d(h_conv2, W_conv3, STRIDE) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 6400])

        # fully connected layer
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc3) + b_fc3


        return s, readout


    def load_networks(self):
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def tensor_board_initial(self):
        if TENSOR_BOARD:
            self.summary_writer = tf.summary.FileWriter("tensorboard", self.sess.graph)
            self.train_cost = SnakeDqn.bias_variable([1])
            tf.summary.scalar('traincost', tf.reduce_sum(self.train_cost))
            self.merged_summary_op = tf.summary.merge_all()

    def tensor_board(self,cost_value):
        if TENSOR_BOARD:
            summary_str = self.sess.run(self.merged_summary_op, feed_dict={self.train_cost: [cost_value]})
            self.summary_writer.add_summary(summary_str, self.t)


    def tack_action(self, readout_t,energy):
        a_t = np.zeros([ACTIONS])
        if random.random() <= (self.epsilon if TRAIN_MODE else TEST_EPSILON):
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            print("------Trained  Action------")
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        if (action_index > 7) and (energy < 30): # chose actions with accelerate but it's not long enough, give up on accelerate
            a_t[action_index] = 0
            action_index -= 8
            a_t[action_index] = 1
        if sum(a_t) != 1:
            raise ValueError('Multiple input actions!')
        move_time0 = SnakeDqn.record_time()
        self.mysnk.Move(MOVE_DICT[action_index][0], MOVE_DICT[action_index][1])
        if action_index > 7:
            self.mysnk.Acc(1)
        else:
            self.mysnk.Acc(0)
        move_time1 = SnakeDqn.record_time()
        if TIME_SWITCH:
            print("Move use time:\t", move_time1 - move_time0,"\tTake Action",action_index )
        return a_t, action_index


    def get_isDead(self):
        get_isDead_Time0 = SnakeDqn.record_time()
        TerminalJson = self.mysnk.GetIsDead()
        if TerminalJson['result']['status'] == 0:
            terminal = TerminalJson['result']['IsDead']
        else:
            print("Get IsDead failed!\n")
            return -1
        get_isDead_Time1 = SnakeDqn.record_time()
        if TIME_SWITCH:
            print("Get is Dead us time:\t", get_isDead_Time1 - get_isDead_Time0)
        return terminal

    def getEnergy(self):
        JsonObj = self.mysnk.GetCurrentEnergy()
        if JsonObj['result']['status'] == 0:
            return JsonObj['result']['Energy']
        else:
            print("GetEnergy failed\n")
            return None

    def back_restart(self):
        self.mysnk.Back()
        print("Back button has been pushed")
        time.sleep(5)
        print("waited for 5 seconds")
        self.last_back_time = time.clock()
        self.mysnk.SinglePlay()
        print("SinglePlay button has been pushed")
        time.sleep(15)
        print("waited for 15 seconds")


    def get_input_new(self,Eng_beforeAction,kill_cnt_before):
        picture_time0 = SnakeDqn.record_time()
        magnitude = int(Eng_beforeAction / 2000)
        map_ratio = min(20, 10 + (magnitude * 2)) * 1.52

        # if Eng_beforeAction > 3000:
        #     map_ratio = 15
        # else:
        #     map_ratio = 10
        all_x_t1, x_t1, Eng_afterAction, kill_cnt_after, isDead = dr.draw_map(self.mysnk,map_ratio)
        picture_time1 = SnakeDqn.record_time()
        if TIME_SWITCH:
            print("Get Infor use time:\t", picture_time1 - picture_time0)
        if isDead:
            print("Player dead!")
            r_t = -12
            x_t1 = []
            kill_cnt_after = 0
            deadtime = time.clock()
            time.sleep(10)  # wait for the end screen to show
            print("Waited for 10 seconds")
            if deadtime - self.last_back_time > 1800:
                print("back and restart")
                self.back_restart()
            else:
                self.mysnk.Restart()
                print("Restart button has been pushed")
                time.sleep(2)  # make sure next loop will obtain correct picture & energy
            Eng_afterAction = self.getEnergy()
            while Eng_afterAction is None:
                Eng_afterAction = self.getEnergy()
        else:
            if DRAW_PICTURE:
                cv2.imshow("src", x_t1)
                cv2.waitKey(1)
            rito = 5.0
            r_t = min((Eng_afterAction - Eng_beforeAction) / rito,11)
            # if 0<r_t<=(3/rito):
            #     r_t = 0.0003
            # r_t = max(r_t, 0.0002) if r_t>=0 else r_t
            kill_num = kill_cnt_after - kill_cnt_before
            if kill_num > 0:
                r_t += kill_num * 12.0
                print("Killed %d snake!" % kill_num)
            # r_t = min(r_t,11)
        return x_t1, r_t, Eng_afterAction, kill_cnt_after, isDead



    def store_transition (self,a_t,r_t,x_t1,s_t,terminal):
        x_t1 = np.reshape(x_t1, (PICTURESIZE, PICTURESIZE, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        if r_t == 0:
            if random.random() < 0.5:
                self.D.append((s_t, a_t, r_t, s_t1, terminal))
        else:
            self.D.append((s_t, a_t, r_t, s_t1, terminal))
            # if r_t > 0.8 : # increase samples of eating large energy such as corpse
            #     times = round(10*(math.log(3 + r_t)))
            #     D.extend([(s_t, a_t, r_t, s_t1, terminal)] * times)
            # if r_t < 0: # increase samples of get killed
            #     D.extend([(s_t, a_t, r_t, s_t1, terminal)] * dead_repeat)
        if len(self.D) > REPLAY_MEMORY:
            self.D.popleft()
        return s_t1


    def train_initial(self, s, readout):
        # define the cost function
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        # (self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,use_locking=False, name="Adam")
        print("Initial Complete!")

        all_x_t, x_t, Eng_beforeAction,kill_cnt_before,isDead = dr.draw_map(self.mysnk,10)


        while isDead:
            all_x_t, x_t, Eng_beforeAction, kill_cnt_before, isDead = dr.draw_map(self.mysnk,10)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 80*80*4

        # saving and loading networks
        self.load_networks()
        self.epsilon = LAST_EPSILON if CONTINUE_TRAIN else INITIAL_EPSILON
        self.t = LAST_T+1 if CONTINUE_TRAIN else 1
        self.dead_repeat = INITIAL_DEAD_REPEAT
        self.tensor_board_initial()
        readout_org = readout.eval(feed_dict={s: [s_t]})
        readout_t = readout_org[0]
        a_t, action_index = self.tack_action(readout_t, Eng_beforeAction)
        return  s_t, readout_t,a_t, action_index,Eng_beforeAction,a,y,cost,kill_cnt_before


    def train_network(self,readout,s,a,y,cost):
        train_time0 = SnakeDqn.record_time()
        if TRAIN_MODE and ((self.t-LAST_T if CONTINUE_TRAIN else self.t) > OBSERVE):
            # sample a minibatch to train on
            minibatch = random.sample(self.D, BATCH)

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
            self.train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
            cost_value = cost.eval(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})
            print ("  / COST_absolute: ", cost_value)
            train_time1 = SnakeDqn.record_time()
            if TIME_SWITCH:
                print("Train Network use time:\t",train_time1-train_time0)
            return cost_value


    def save_progress(self, frequence):
        if TRAIN_MODE and (self.t % frequence == 0):
            self.saver.save(self.sess, 'saved_networks/' + GAME + '-dqn', global_step = self.t)
            print ("Network Saved!")


    def print_info(self,action_index,r_t,readout_t,kill_cnt_after):
        if self.t <= OBSERVE:
            state = "observe"
        elif OBSERVE < self.t <=  int(OBSERVE + EXPLORE):
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", self.t, "/ STATE", state, \
              "/ EPSILON", (self.epsilon if TRAIN_MODE else TEST_EPSILON), "/ ACTION", action_index, "/ REWARD", r_t, \
              "/ Q_MAX %e" % np.max(readout_t),"/ KILL_CNT", kill_cnt_after)



    def run_circulation(self, s, readout):
        s_t, readout_t, a_t, action_index, Eng_beforeAction,a,y,cost,kill_cnt_before = self.train_initial(s, readout)
        while "crazy snake" != "dead snake":
            loop_time0 = SnakeDqn.record_time()
            '''
            for i in range(32):
                cv2.imshow(str(i), hidden1[0][:][:][i])
            '''
            try:
                # a_t, action_index= self.tack_action(readout_t, Eng_beforeAction)
                picture, r_t,Eng_afterAction, kill_cnt_after, terminal = self.get_input_new(Eng_beforeAction,kill_cnt_before)
                # print ("Eng_beforeAction:\t",Eng_beforeAction,"\tEng_afterAction:\t",Eng_afterAction)
                if picture is None: # something wrong, get picture failed
                    print("Infor get failed! \n")
                    continue
                elif picture == []: # dead, do nothing, keep x_t1 as last iteration
                    pass
                else: #alive, obtain new x_t1
                    x_t1 = picture
                s_t1 = self.store_transition(a_t,r_t,x_t1,s_t,terminal)
                self.print_info(action_index,r_t,readout_t,kill_cnt_after)

                readout_org = readout.eval(feed_dict={s: [s_t1]})
                readout_t = readout_org[0]
                a_t, action_index = self.tack_action(readout_t, Eng_afterAction)

                cost_value = self.train_network(readout, s, a, y, cost)
                self.tensor_board(cost_value)
                self.save_progress(5000)
            except:
                traceback.print_exc()
                print("Unexcepted error happend,epsilon = ", self.epsilon)
                break

            Eng_beforeAction = Eng_afterAction
            kill_cnt_before = kill_cnt_after
            s_t = s_t1
            self.t += 1
            if self.epsilon > FINAL_EPSILON and self.t > OBSERVE:
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            loop_time1 = SnakeDqn.record_time()
            if TIME_SWITCH:
                print("One loop use time:\t", loop_time1 - loop_time0, "\n")


    def playGame(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.InteractiveSession(config=config)
        s, readout = SnakeDqn.createNetwork()
        self.run_circulation(s, readout)

def main():
    snake_dqn = SnakeDqn()
    snake_dqn.playGame()

if __name__ == "__main__":
    main()