import os, sys, subprocess
import time
import traci
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

scenes = ["straight", "left_turn", "right_turn"]
step_size = 0.1
plot_flag = False

# For each scenario, run SUMO and record [time, x, y, v_x, v_y] at every time step
for choice in range(len(scenes)):
    file_name = "ref_trajectory_" + scenes[choice] + ".csv"
    df = pd.DataFrame(columns=['x', 'y', 'v', 'angle'], dtype='float')
    step = 0
    max_step = 1000
    row_ind = 0
    l_steps, l_v = [], []

    sumoCmd = ["sumo", "-c", "config_files/4way_" + scenes[choice] + "_ref.sumocfg", "-S", "-Q", '--step-length', str(step_size)]
    traci.start(sumoCmd)

    while step < max_step:
        traci.simulationStep()
        step += 1

        if not traci.vehicle.getIDList():
            continue
        else:

            # get measurements from SUMO
            for carID in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(carID)

                ''' The angle from SUMO does not match our assumptions about x-y coordinates; it differs by 90 degrees.
                    We subtract the angle from 90 to match our assumption'''
                angle = math.radians(90 - traci.vehicle.getAngle(carID))
                v = traci.vehicle.getSpeed(carID)
                v_x = v * math.cos(angle)
                v_y = v * math.sin(angle)
                ts = (step - 1) * step_size
                df.loc[row_ind] = [x, y, v, angle]
                row_ind += 1
                l_steps.append(step - 1)
                l_v.append(v)
        # time.sleep(0.2)

    traci.close()

    # write the dataframe to a tab-separated CSV file
    df.to_csv(file_name, sep=',', index=False)

    # velocity-vs-time plot? If yes, set "plot_flag = True"
    if plot_flag:
        plt.plot(l_steps, l_v)
plt.show()