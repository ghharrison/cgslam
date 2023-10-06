import pandas
import re
import numpy as np
import math

import os

FILETYPE_COLNAME_MAPPING = {
    "Robot_Measurement.dat": ["Time", "Subject", "Range", "Bearing"],
    "Robot_Groundtruth.dat": ["Time", "x", "y", "Orientation"],
    "Robot_Odometry.dat": ["Time", "Velocity", "Angular Velocity"],
    "Landmark_Groundtruth.dat": ["Subject", "x", "y", "x std-dev", "y std-dev"],
    "Barcodes.dat": ["Subject #", "Barcode #"],
}

####################################################
# File utilities
####################################################
def load_mrclam_file(dataset_dir, filename):

    # Remove robot number from filename if applicable
    generic_filename = "".join([x for x in filename if not x.isdigit()])
    print("generic filename: ", generic_filename)
    combined_filename = os.path.join(dataset_dir, filename)
    if generic_filename not in FILETYPE_COLNAME_MAPPING.keys():
        return False

    return pandas.read_csv(
        combined_filename,
        delim_whitespace=True,
        names=FILETYPE_COLNAME_MAPPING[generic_filename],
        comment='#',
    )#.to_numpy()

def round_measurements(measurements: pandas.DataFrame, timesteps, sample_time):
    nr, nc = measurements.shape # #rows, #columns of original data
    
    new_data = pandas.DataFrame(data=measurements, columns=measurements.columns)
    new_data = new_data.rename({'Time': 'Timestep'}, axis=1)   
    new_data.Timestep = np.round(new_data.Timestep / sample_time).astype(int)

    return new_data



def interpolate(old_data, name, timesteps, sample_time, max_time):
    nr, nc = old_data.shape # #rows, #columns of original data

    # TODO: Find out why it thinks ground truth is 8 columns

    k = 0 # current row (new data)
    t = 0 # time (NOT timestep)
    i = 0 # current row (old data)
    p = 0 # timestep time amount...?



    new_data = pandas.DataFrame(data=np.zeros((int(timesteps) + 1, nc)), columns=old_data.columns)

    while k <= timesteps: # Keep going until all times seen
        if not k % 5000:
            print(f"t:{t}\tk:{k}\ti:{i}\tp:{p}")
        new_data.iloc[k, 0] = t # Set the current row's time to the current time

        if not k and old_data.iloc[i, 0] == 0:
            new_data.iloc[0, :] = old_data.iloc[0, :]
        # Advance i until it catches up to/goes past the new data's row (or reaches the last row)
        while old_data.iloc[i, 0] <= t:
            if i == nr - 1:
                break
            i += 1

        # If i is the first or last timestep and this is odometry,
        # Override the velocity and angular velocity to 0
        # ohterwise copy over the existing value
        if i == 0 or i == nr - 1:
            if 'Odo' not in name:
                # print(name, " is copying over for ", i, " ", k)
                new_data.iloc[k, 1:] = old_data.iloc[i, 1:]
            else:
                # print(name, "is NOT copying over for ", i, " ", k)
                new_data.iloc[k, 1:] = 0

        else:
            # p is (time - old data previous row time) / (old data time gap between this and last row)
            p = (t - old_data.iloc[i-1, 0]) / (old_data.iloc[i,0] - old_data.iloc[i-1,0])
            if nc == 8: # i.e. ground truth data
                sc = 2 # "start column"...?
                new_data.iloc[k, 1] = old_data.iloc[i, 1] # keep ID number
            else:
                sc = 1

            for c in range(sc, nc):
                if nc == 8 and c >=5:
                    d = old_data.iloc[i, c] - old_data.iloc[i-1, c] # GT orientation in radians (0-2pi)
                    
                    # Normalize to the range (-pi, pi)
                    if d > math.pi:
                        d = d - 2*math.pi
                    elif d < -math.pi:
                        d = d + 2*math.pi

                    new_data.iloc[k, c] = p*d + old_data.iloc[i-1, c]
                else:
                    new_data.iloc[k, c] = p*(old_data.iloc[i, c] - old_data.iloc[i-1, c]) + old_data.iloc[i-1, c]

        k += 1
        t += sample_time
        # t = np.round(t, decimals=2) # assuming timestep is not smaller than 0.01

    print(new_data.head())
    print("...")
    print(new_data.tail())
    print("Final t: ", t)
    return new_data

def adjust_mrclam_dataset(n_robots, robot_gts, sample_time):
    # first and last row time
    min_time=robot_gts[0].iloc[0]["Time"]
    max_time=robot_gts[0].iloc[-1]["Time"]

    # "Normalize" the timestamps by setting them to start from 0
    for j in range(0, n_robots):
        robot_gts[j]["Time"] = robot_gts[j]["Time"] - min_time

    max_time = max_time - min_time
    timesteps = np.floor(max_time/sample_time) + 1

    return robot_gts




def sample_mrclam_dataset(n_robots, robot_gts, robot_measurements, robot_odometries, sample_time):
    """
    After loading all the MR-CLAM files, sample them by changing from
    absolute (epoch) time to timesteps {sample_time} apart. 
    This should be done once.
    """
    # first and last row time
    min_time=robot_gts[0].iloc[0]["Time"]
    max_time=robot_gts[0].iloc[-1]["Time"]

    for i in range(1, n_robots):
        min_time = min(min_time, robot_gts[i].iloc[0]["Time"])
        max_time = max(max_time, robot_gts[i].iloc[-1]["Time"])

    # "Normalize" the timestamps by setting them to start from 0
    for j in range(0, n_robots):
        robot_gts[j]["Time"] = robot_gts[j]["Time"] - min_time
        robot_measurements[j]["Time"] = robot_measurements[j]["Time"] - min_time
        robot_odometries[j]["Time"] = robot_odometries[j]["Time"] - min_time

    max_time = max_time - min_time
    timesteps = np.floor(max_time/sample_time)+ 1

    print(min_time, " IS  TIMESTEP 0 (t=0)")
    print("SAMPLING TIME IS ", sample_time)
    print("TOTAL NUMBER OF TIMESTEPS: ", timesteps)

    # "Sample" (interpolate) the data
    for z in range(0, n_robots):

        robot_gts[z] = interpolate(robot_gts[z], f"Robot{z}_Groundtruth.dat", timesteps, sample_time, max_time)
        # robot_measurements[z] = interpolate(robot_measurements[z], f"Robot{z}_Measurement.dat", timesteps, sample_time, max_time)
        robot_measurements[z] = round_measurements(robot_measurements[z], timesteps, sample_time)
        print(robot_measurements[z].head())
        print("...")
        print(robot_measurements[z].tail())
        robot_odometries[z] = interpolate(robot_odometries[z], f"Robot{z}_Odometry.dat", timesteps, sample_time, max_time)


    return timesteps, robot_gts, robot_measurements, robot_odometries

def create_barcode_mapping(barcodes_df):
    mapping = {}
    for i in range(len(barcodes_df)):
        mapping[barcodes_df.iloc[i]["Barcode #"]] = barcodes_df.iloc[i]["Subject #"]

    return mapping
