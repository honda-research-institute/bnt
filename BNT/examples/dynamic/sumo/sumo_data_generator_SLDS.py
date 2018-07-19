import traci
import numpy as np
import math
import csv
from ekf import ekf_nonlin as EKF
import matplotlib.pyplot as plt


def ekf_plots():
    # x vs time
    plt.subplot(321)
    plt.plot(ts, x_meas[:, 0], 'k+', label='noisy measurements')
    plt.plot(ts, x_true[:, 0], color='b', label=scenes[choice] + ' truth')
    plt.plot(ts, x_pred[:, 0], 'r', label='ekf prediction')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(loc=2)
    # y vs time
    plt.subplot(322)
    plt.plot(ts, x_meas[:, 1], 'k+', label='noisy measurements')
    plt.plot(ts, x_true[:, 1], color='b', label=scenes[choice] + ' truth')
    plt.plot(ts, x_pred[:, 1], 'r', label='ekf prediction')
    plt.xlabel('time')
    plt.ylabel('y')
    # plt.ylim(180, 220)
    plt.legend(loc=2)
    # x vs y
    plt.subplot(323)
    plt.plot(x_meas[:, 0], x_meas[:, 1], 'k+', label='noisy measurements')
    plt.plot(x_true[:, 0], x_true[:, 1], color='b', label='truth value')
    plt.plot(x_pred[:, 0], x_pred[:, 1], 'r', label='ekf prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2)
    plt.xlim(5, 105)
    plt.axis('equal')
    # v vs time
    plt.subplot(324)
    plt.plot(ts, x_meas[:, 2], 'k+', label='noisy measurements')
    plt.plot(ts, x_true[:, 2], color='b', label=scenes[choice] + ' truth')
    plt.plot(ts, x_pred[:, 2], 'r', label='ekf prediction')
    plt.xlabel('time')
    plt.ylabel('v')
    # plt.ylim(0,30)
    plt.legend(loc=4)
    # heading vs time
    plt.subplot(325)
    plt.plot(ts, map(math.degrees, x_true[:, 3]), color='b', label=scenes[choice] + ' truth')
    plt.plot(ts, map(math.degrees, x_pred[:, 3]), 'r', label='ekf prediction')
    plt.xlabel('time')
    plt.ylabel('heading')
    plt.legend(loc=4)
    plt.show()


def normalize_feature(x, cur_min, cur_max, new_min=-1, new_max=1):
    """ Scale x to [new_min, new_max] from [cur_min, cur_max] """
    normalized_x = (new_max - new_min) * (x - cur_min) / (cur_max - cur_min) + new_min
    return normalized_x

np.random.seed(0)
num_run = 10
scenes = ['straight', 'right_turn']
step_size = 0.1
noise_std = 0.8
intent_dict = {'straight' : {'A':1, 'P':2, 'L':3}, 'right_turn': {'A':4, 'P':5, 'L':6}}
# data_file = open('data_'+str(num_run)+'.csv','wb')
data_file = open('30/data_'+str(num_run)+'.csv','wb')
data_file_normal = open('30/data_'+str(num_run)+'_normal.csv','wb')
data_file_truth = open('30/data_truth_'+str(num_run)+'.csv','wb')
label_file = open('30/label_'+str(num_run)+'.csv','wb')


writer = csv.writer(data_file)
writer_normal = csv.writer(data_file)

intr_size = 10.0
twice_intr = 2 * intr_size
x_intr = 200.0
y_intr = 200.0
start_prediction = 170 #x_intr-twice_intr

min_v,max_v = 10.0, 30.0
min_x, max_x = start_prediction-x_intr, twice_intr
min_y,max_y = -twice_intr, 0.0
min_heading, max_heading = -1.57, 0.0

for i in range(num_run):
    x_true, x_meas, x_pred, x_pred_normal, labels, ts = [], [], [], [], [], []
    ekf = EKF.EKF()
    step = 0
    max_step = 1000
    l_steps, l_v = [], []
    ekf_initialized = True
    t_40m, t_50m = 0, 0
    captured_40m, captured_50m = False, False
    x_40m, x_50m = 0, 0
    initial_v = 0
    prev_angle = 0.0
    choice = np.random.randint(len(scenes))
    sumoCmd = ["sumo", "-c", "config_files/4way_" + scenes[choice] + ".sumocfg", "-S", "-Q", '--step-length', str(step_size), '--seed', str(np.random.randint(0,10000))]
    traci.start(sumoCmd)
    while step < max_step:
        traci.simulationStep()
        step += 1
        t_val = step * step_size

        if not traci.vehicle.getIDList():
            continue
        else:

            # get measurements from SUMO
            for carID in traci.vehicle.getIDList():
                # print traci.vehicle.getSpeedFactor(carID)
                x, y = traci.vehicle.getPosition(carID)

                ''' The angle from SUMO does not match our assumptions about x-y coordinates; it differs by 90 degrees.
                    We subtract the angle from 90 to match our assumption'''
                angle = math.radians(90 - traci.vehicle.getAngle(carID))
                v = traci.vehicle.getSpeed(carID)
                vx = v * math.cos(angle)
                vy = v * math.sin(angle)
                acc = traci.vehicle.getAccel(carID)
                delta = (prev_angle-angle)/step_size
                if (start_prediction) < x < (x_intr + twice_intr) and \
                        (y_intr - twice_intr) < y < (y_intr + twice_intr):
                    cur_x = [x, vx, y, vy]
                    cur_x[0] = cur_x[0] - x_intr
                    cur_x[2] = cur_x[2] - y_intr
                    x_true.append(cur_x)
                    if choice == 0:
                        if x < -intr_size:
                            labels.append(1)
                        elif -intr_size < x < intr_size and -intr_size < y < intr_size:
                            labels.append(2)
                        else:
                            labels.append(3)
                    else:
                        if x < -intr_size:
                            labels.append(4)
                        elif -intr_size < x < intr_size and -intr_size < y < intr_size :
                            labels.append(5)
                        else:
                            labels.append(6)
                    ts.append(t_val)
                # # if (start_prediction) < x < (x_intr + twice_intr) and \
                # #         (y_intr - twice_intr) < y < (y_intr + twice_intr):
                # #     cur_x = [x, y, v, angle]
                # #
                # #     # Add noise to current measurements
                # #     cur_x_meas = cur_x[:-1] + np.random.normal(0, noise_std, len(cur_x) - 1)
                # #     # # We want to start prediction at x = 50 meters
                # #     # if x > (start_prediction-10) and not captured_40m:
                # #     #     t_40m = t_val
                # #     #     x_40m = x
                # #     #     captured_40m = True
                # #     # elif start_prediction < x < (x_intr + twice_intr):
                # #     #     if not captured_50m:
                # #     #         t_50m = t_val
                # #     #         x_50m = x
                # #     #         captured_50m = True
                # #     #         initial_v = (x_50m - x_40m) / (t_50m - t_40m)
                # #     #     if ekf_initialized:
                # #     #         # print 'initial velocity: ' + str(initial_v)
                # #     #         initial_state = np.array([cur_x_meas[0], cur_x_meas[1], 0.0, initial_v, 0.0, 0.0])
                # #     #         ekf.reset(np.array(initial_state))
                # #     #         ekf_initialized = False
                # #
                # #     # print 'first_step: ' + str(ekf_initialized)
                # #     if not ekf_initialized:
                # #         x_meas.append(cur_x_meas)
                # #         cur_x[0] = cur_x[0] - x_intr
                # #         cur_x[1] = cur_x[1] - y_intr
                # #         cur_x = cur_x + [acc, delta]
                # #         x_true.append(cur_x)
                # #
                # #         # EKF prediction
                # #         dt = step_size
                # #         ekf.predict(dt)
                # #
                # #         # EKF update
                # #         z = [cur_x_meas[0], cur_x_meas[1]]
                # #         ekf.update(z)
                # #         ekf.xlist_reordered[0] = ekf.xlist_reordered[0] - x_intr
                # #         ekf.xlist_reordered[1] = ekf.xlist_reordered[1] - y_intr
                # #
                # #         x,y,v,heading= \
                # #             ekf.xlist_reordered[0], ekf.xlist_reordered[1], ekf.xlist_reordered[2], ekf.xlist_reordered[3]
                # #
                # #         x_norm, y_norm, v_norm, heading_norm =\
                # #         normalize_feature(x, min_x, max_x, -1, 1), normalize_feature(y, min_y, max_y, -1, 1), \
                # #         normalize_feature(v, min_v, max_v, -1, 1), normalize_feature(heading, min_heading, max_heading, -1, 1)
                #
                #         x_pred_normal.append (np.array([x_norm, y_norm, v_norm, heading_norm]))
                #         x_pred.append(ekf.xlist_reordered)
                #         if choice == 0:
                #             if x < -intr_size:
                #                 labels.append(1)
                #             elif -intr_size < x < intr_size and -intr_size < y < intr_size:
                #                 labels.append(2)
                #             else:
                #                 labels.append(3)
                #         else:
                #             if x < -intr_size:
                #                 labels.append(4)
                #             elif -intr_size < x < intr_size and -intr_size < y < intr_size :
                #                 labels.append(5)
                #             else:
                #                 labels.append(6)
                #         ts.append(t_val)

    traci.close()
    x_meas, x_true, x_pred, x_pred_normal, labels = \
        np.array(x_meas), np.array(x_true), np.array(x_pred), np.array(x_pred_normal), np.array(labels)
    ts =np.array(ts)

    # ekf_plots()

    # # write the dataframe to a tab-separated CSV file
    # x_pred = x_pred.flatten()
    # x_pred.tofile(data_file, sep=',')
    # data_file.write('\n')
    #
    # x_pred_normal = x_pred_normal.flatten()
    # x_pred_normal.tofile(data_file_normal, sep=',')
    # data_file_normal.write('\n')

    x_true = x_true.flatten()
    x_true.tofile(data_file_truth, sep=',')
    data_file_truth.write('\n')

    labels.tofile(label_file, sep=',')
    label_file.write('\n')

# data_file.close()
# data_file_normal.close()
data_file_truth.close()
label_file.close()



