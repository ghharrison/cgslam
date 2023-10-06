import copy
from pathlib import Path
import time
from typing import List

import imageio
import slam_utils
import numpy as np
import argparse
import pandas as pd
import os
import itertools

import matplotlib.pyplot as plt

from python_graphslam.graph import Graph
from python_graphslam.vertex import Vertex
from python_graphslam.pose.r2 import PoseR2
from python_graphslam.pose.se2 import PoseSE2
from python_graphslam.edge.edge_odometry import EdgeOdometry

COMM_RANGE = 2

GT_ODOMETRY_NOISE_STDEV = np.array([0.0025, 0.0025, 0.0001])
GT_LANDMARK_NOISE_STDEV = np.array([1, 1, np.pi/2])

RELATIVE_RANGE_ERROR_TOLERANCE = 1
RELATIVE_BEARING_ERROR_TOLERANCE = np.pi / 8
############################################################
class GlobalSimController:
    """
    This is a global SIMULATION controller, not a robot controller.
    While it handles communications and makes the robots step,
    it does not give them instructions or anything.
    """
    def __init__(self, robots, robot_gts, timesteps, args):
            self.robots = robots
            self.robot_gts = robot_gts
            self.timesteps = timesteps
            self.neighbor_bots = [] # Tuples of robot #s of nearby robots (< 2m)
            self.t = 0
            self.step_length = args.sample_time

            self.one_sees_two = []
            self.two_sees_one = []

            self.args = args

            self.gif_frames = [] # list of list
            for i in range(len(robots)):
                self.gif_frames.append([])



    def execute(self):
        for t in range(int(self.timesteps)):
            self.t = t
            if not t % 1000:
                print(f"---Performing timestep {t}---")
            for robot in self.robots:
                robot.step()

                if t > 0 and self.args.opt_interval is not None and t % self.args.opt_interval == 0:
                    print("Optimizing map for robot ", robot.robot_num)
                    robot.total_opt_time += robot.optimize(self.args.solver)
                    robot.pose = robot.last_odometry_vertex.pose

                if self.args.save_interval is not None and t % self.args.save_interval == 0:
                    savepath = args.dataset_dir + f'/saved_map_{robot.robot_num}' + '/' + f'map_{robot.robot_num}_{self.t}.png'
                    print(f"Saving robot {robot.robot_num} map to {savepath}")
                    robot.graph.plot(title=f"Map from robot {robot.robot_num}, t={t} timesteps", savepath=savepath)
                    loaded_plot = imageio.v2.imread(savepath)
                    self.gif_frames[robot.robot_num - 1].append(loaded_plot)
                

            if not self.args.no_share:
                self.handle_comms()

        # print("1 sees 2: ", self.one_sees_two)
        #print("2 sees 1: ", self.two_sees_one)
        # print("Sees each other: ", list(set(self.one_sees_two) & set(self.two_sees_one)))
        total_opt_time = 0.    
        for robot in self.robots:
            if not self.args.no_opt:
                robot.total_opt_time += robot.optimize(self.args.solver)
            total_opt_time += robot.total_opt_time # don't do this if we use parallelism
            print(f"*** Robot {robot.robot_num} spent in total {robot.graph.total_solve_time}s solving linear equations***")
            print(f"*** Robot {robot.robot_num} spent in total {robot.total_opt_time}s optimizing maps***")
            if self.args.record:
                gif_savepath = args.dataset_dir + f'/saved_map_{robot.robot_num}' +'/' + f'animated_map_{robot.robot_num}.gif'
                imageio.mimsave(gif_savepath, self.gif_frames[robot.robot_num-1], fps=(60/self.args.save_interval))
        print(f"***ALL ROBOTS SPENT COMBINED {total_opt_time}s OPTIMIZING***")

        

    def handle_comms(self):
        # Get all pairs of robots
        for robot, other_robot in itertools.combinations(self.robots, 2):
            _, x, y, orientation = robot.gt.iloc[self.t] # remember gt is a Pandas data frame
            robot_gt_pose = PoseSE2([x, y], orientation)
            _, x, y, orientation = other_robot.gt.iloc[self.t] # remember gt is a Pandas data frame
            other_robot_gt_pose = PoseSE2([x, y], orientation)
            gt_dx = other_robot_gt_pose[0] - robot_gt_pose[0]
            gt_dy = other_robot_gt_pose[1] - robot_gt_pose[1]

            # Only attempt map sharing if robots can see each other
            if robot.robot_num == 1 and 2 in robot.current_visible_robots:
                self.one_sees_two.append(self.t)
            if other_robot.robot_num == 2 and 1 in other_robot.current_visible_robots:
                self.two_sees_one.append(self.t)


            can_see = (robot.robot_num in other_robot.current_visible_robots) and (other_robot.robot_num in robot.current_visible_robots)

            if np.sqrt(gt_dx ** 2 + gt_dy ** 2) > COMM_RANGE or not can_see:
                # Remove from neighbor set
                if (robot.robot_num, other_robot.robot_num) in self.neighbor_bots:
                    self.neighbor_bots.remove((robot.robot_num, other_robot.robot_num))
                    # print(f"Removing robots {robot.robot_num}/{other_robot.robot_num} as neighbors")
                continue

            # Check that they're not already neighbors (should share map when first meeting)
            if (robot.robot_num, other_robot.robot_num) in self.neighbor_bots:
                continue

            self.neighbor_bots.append((robot.robot_num, other_robot.robot_num))

            # Share maps
            print(self.t, ": Sharing maps for robots ", robot.robot_num, "/", other_robot.robot_num)
            self.share_maps(robot, other_robot)


    def share_maps(self, robot, other_robot):
        robot.send_map(other_robot)
        other_robot.send_map(robot)




############################################################
class GraphSLAMRobot:
    def __init__(self) -> None:
        pass

    def add_new_odometry_vertex(self):
        raise NotImplementedError

    def update_landmarks(self, landmark_obs):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def send_map(self, other_robot):
        raise NotImplementedError
    
    def optimize(self, solver=None):
        # Optimize graph to minimize error
        # WARNING: Doing this "fixes" the first vertex so it stays at (0, 0). 
        # We may not want this for when we share the vertex, or else 
        # multiple vertices will be fixed and it'll get weird.
        opt_start_time = time.time()
        self.graph.optimize(fix_first_pose=True, solver=solver)
        time_taken = time.time() - opt_start_time
        print(f"Robot {self.robot_num} finished optimizing map in {time_taken} seconds")
        return time_taken

############################################################

class TrivialGTOnlyRobot(GraphSLAMRobot):
    """
    Will use only ground truth info of robots and landmarks, resulting in hopefully
    an exactly true map that needs no optimization.

    Used for proof of concept in map merging
    """
    def __init__(self, robot_num, gt: pd.DataFrame, landmark_gt: pd.DataFrame, measurements: pd.DataFrame, barcode_subject_dict) -> None:
        self.timestep = 0
        self.time = 0 # real time
        self.robot_num = robot_num
        self.gt = gt
        self.landmark_gt = landmark_gt
        self.measurements = measurements
        self.barcode_subject_dict = barcode_subject_dict

        self.dx_errors = []
        self.dy_errors = []
        self.w_errors = []

        # self.current_position_v_idx = -1
        self.last_odometry_vertex = None

        # Read the first line from ground truth and initialize the graph and position 
        _, x, y, orientation = gt.iloc[0] # remember gt is a Pandas data frame

        self.pose = PoseSE2([x, y], orientation)


        # Initialize a graph with one vertex and no edges
        self.graph = Graph([], [])
        self.add_new_odometry_vertex()

        # TODO: Register any landmarks in the graph that can be seen from the initial position
        self.current_visible_robots = []
        self.seen_landmarks = set()
        self.detected_all_lm_time = (None, float("inf")) # timestep, time

    def add_new_odometry_vertex(self):         
        new_vertex = Vertex(f"{self.robot_num}_{self.timestep}", self.pose)        
        self.graph._vertices.append(new_vertex)

        # Only add edges if this is not the first vertex
        if len(self.graph._vertices) != 1:
            new_odometry_edge = EdgeOdometry(
                vertex_ids=[self.last_odometry_vertex.id, new_vertex.id],
                information=np.eye(3), # probably a bad idea
                estimate=(new_vertex.pose - self.last_odometry_vertex.pose),
                vertices=[self.last_odometry_vertex, new_vertex],
            )
            self.graph._edges.append(new_odometry_edge)
        self.last_odometry_vertex = self.graph._vertices[-1]
        self.graph._link_edges()

    def update_landmarks(self, landmark_obs):
        # If the landmark is not already in the graph, add it.
        # Then add all edges

        for _, obs in landmark_obs.iterrows():
            # Get the subject# from the barcode number
            try:
                landmark_subject_num = self.barcode_subject_dict[obs["Subject"]]
            except Exception:
                continue

            # disregard other robots for now
            if landmark_subject_num not in range(6, 21):
                continue

            # global frame pose (ground truth)
            r = self.landmark_gt.loc[self.landmark_gt["Subject"] == landmark_subject_num]

            _, gt_lx, gt_ly, _, _ = r.squeeze()
            gt_pose = PoseSE2([gt_lx, gt_ly], 0)
            # gt_pose.orientation = 0 # Landmarks have no orientation for our purposes

            # exit()
            if str(landmark_subject_num) not in self.seen_landmarks:
                self.seen_landmarks.add(str(landmark_subject_num))

                # create a vertex for the new landmark
                # (Will use the landmark's subject number as vertex ID since it should be unique)
                landmark_vertex = Vertex(str(landmark_subject_num), gt_pose)
                self.graph._vertices.append(landmark_vertex)

            else:
                pass #landmark_vertex = next(v for v in self.graph._vertices if v.id == str(landmark_subject_num)) # how can we retrieve a particular landmark's vertex?
        
            # TODO: Should the measurement estimate be in local or global frame?
            new_landmark_edge = EdgeOdometry(
                vertex_ids=[self.last_odometry_vertex.id, str(landmark_subject_num)],
                information=np.eye(3),
                estimate=gt_pose - self.pose,
                # estimate=PoseSE2([lf_x, lf_y], landmark_bearing),
                # vertices=[self.last_odometry_vertex, landmark_vertex],
            )
            self.graph._edges.append(new_landmark_edge)
        self.graph._link_edges()

    def update_current_visible_robots(self, current_obs):
        """
        Get a list of the robots (not landmarks) the robot
        can see at the current timestep. 
        Used by the global sim controller to decide whether
        robots should communicate (if they can't see each other,
        it won't be possible to reach a consensus on relative position.)
        """
        visible_robots = []
        if current_obs.empty:
            return
        
        for _, obs in current_obs.iterrows():
            # Get the subject# from the barcode number
            # print("Subject: ", obs["Subject"])
            try:
                subject_num = self.barcode_subject_dict[obs["Subject"]]
            except Exception:
                print("Ignoring invalid observation:\n", obs)

            # only consider robots
            if subject_num not in range(1, 6):
                continue


            if subject_num not in visible_robots:
                visible_robots.append(subject_num)

        self.current_visible_robots = visible_robots


    def send_map(self, other_robot):
        # Add all of "our" vertices not in "their" graph to their graph
        old_other_v_len = len(other_robot.graph._vertices)
        for v in self.graph._vertices:
            if v.id not in [other.id for other in other_robot.graph._vertices]:
                other_robot.graph._vertices.append(v)

                # Update the other robot's "seen landmarks" set
                if "_" not in v.id:
                    # Is landmark
                    if v.id not in other_robot.seen_landmarks:
                        # print(v.id, " is not in ", other_robot.seen_landmarks)
                        print("Sharing landmark ", v.id, f" {self.robot_num} --> {other_robot.robot_num}")
                        other_robot.seen_landmarks.add(v.id)

        print("Extended robot ", other_robot.robot_num, f"graph: {old_other_v_len}/{len(other_robot.graph._vertices)} vertices")
        
        old_other_e_len = len(other_robot.graph._vertices)
        for e in self.graph._edges:
            if e not in other_robot.graph._edges:
                other_robot.graph._edges.append(e)

        print("Extended robot ", other_robot.robot_num, f"graph: {old_other_e_len}/{len(other_robot.graph._edges)} edges")

        other_robot.graph._link_edges()

        # Note: NO TRANSLATION is being done because all should exist in global frame
        

    def step(self):
        """
        Advance a single step in the simulation. 
        This will involve reading the robot's next ground truth position
        instead of computing it from odometry information as would usually be done.

        Create a new node and edge representing the step movement, and read the
        corresponding range-bearing measurements to create new landmark nodes
        or new edges for existing landmarks as needed.

        Lastly, if other robots are seen in range, attempt to communicate.
        """
        self.timestep += 1
        self.time = self.gt.iloc[self.timestep]["Time"]

        _, x, y, orientation = self.gt.iloc[self.timestep]
        self.pose = PoseSE2([x, y], orientation)

        self.add_new_odometry_vertex() # odometry vertex?

        # TODO: Spot any landmarks
        # Get all measurements from the current time (there could be many)
        current_obs = self.measurements.loc[self.measurements["Timestep"] == self.timestep]
        if not current_obs.empty:
            # print("Spotted landmark(s) at timestep ", self.timestep)
            self.update_landmarks(current_obs)
            self.update_current_visible_robots(current_obs)

                        # If all landmarks seen, record time(step)
        if len(self.seen_landmarks) == 15 and self.detected_all_lm_time == (None, float("inf")):
            print(self.robot_num, " detected all landmarks at time ", self.time)
            self.detected_all_lm_time = (self.timestep, self.time)

        

############################################################## 
class NoisyGTRobot(TrivialGTOnlyRobot):
    """
    A robot that also uses GT info for odometry and observations,
    but perturbed by some random noise (so optimization is needed).
    """

    def add_pose_noise(self, pose: PoseSE2, scale=GT_ODOMETRY_NOISE_STDEV):
        pose_arr = pose.to_array()
        
        noise = np.random.normal(loc=np.zeros(3), scale=scale)

        noisy_pose = pose_arr + noise

        return PoseSE2([noisy_pose[0], noisy_pose[1]], noisy_pose[2])
    
    def add_new_odometry_vertex(self):
        noisy_pose = self.add_pose_noise(self.pose)         
        new_vertex = Vertex(f"{self.robot_num}_{self.timestep}", noisy_pose)        
        self.graph._vertices.append(new_vertex)

        # Only add edges if this is not the first vertex
        if len(self.graph._vertices) != 1:
            new_odometry_edge = EdgeOdometry(
                vertex_ids=[self.last_odometry_vertex.id, new_vertex.id],
                information=np.linalg.inv(np.diag(GT_ODOMETRY_NOISE_STDEV)), # probably a bad idea
                estimate=(new_vertex.pose - self.last_odometry_vertex.pose),
                vertices=[self.last_odometry_vertex, new_vertex],
            )
            self.graph._edges.append(new_odometry_edge)
        self.last_odometry_vertex = self.graph._vertices[-1]
        self.graph._link_edges()


    def update_landmarks(self, landmark_obs):
        for _, obs in landmark_obs.iterrows():
            try:
                landmark_subject_num = self.barcode_subject_dict[obs["Subject"]]
            except Exception:
                print("Ignoring invalid observation:\n", obs)
                continue

            # disregard other robots for now
            if landmark_subject_num not in range(6, 21):
                continue

            # print("Landmark observation: \n", obs)
            est_range = obs["Range"]
            bearing = obs["Bearing"]

            est_relative_lx = np.cos(bearing) * est_range
            est_relative_ly = np.sin(bearing) * est_range
            est_relative_pose = PoseSE2([est_relative_lx, est_relative_ly], -self.pose[2])
            composed_pose = self.pose + est_relative_pose
            est_global_lx = composed_pose[0]
            est_global_ly = composed_pose[1]

            global_landmark_pose = PoseSE2([est_global_lx, est_global_ly], 0)
            est_measurement_pose_diff = PoseSE2([est_global_lx - self.pose[0], est_global_ly - self.pose[1]], -self.pose[2])

            r = self.landmark_gt.loc[self.landmark_gt["Subject"] == landmark_subject_num]
            _, gt_lx, gt_ly, _, _ = r.squeeze()
            gt_pose = PoseSE2([gt_lx, gt_ly], 0)

            noisy_landmark_pose = self.add_pose_noise(gt_pose, GT_LANDMARK_NOISE_STDEV)
            landmark_vertex = None
            if str(landmark_subject_num) not in self.seen_landmarks:
                self.seen_landmarks.add(str(landmark_subject_num))
                landmark_vertex = Vertex(str(landmark_subject_num), global_landmark_pose)
                self.graph._vertices.append(landmark_vertex)
            new_landmark_edge = EdgeOdometry(
                vertex_ids=[self.last_odometry_vertex.id, str(landmark_subject_num)],
                information=np.linalg.inv(np.diag(GT_LANDMARK_NOISE_STDEV)),
                estimate=est_relative_pose,
                # estimate=gt_pose - self.pose,
                # estimate=PoseSE2([lf_x, lf_y], landmark_bearing),
                vertices=[self.last_odometry_vertex, landmark_vertex] if landmark_vertex else None,
            )
            self.graph._edges.append(new_landmark_edge)
            self.graph._link_edges()

############################################################## 
class OdometryAndMeasurementRobot(NoisyGTRobot):
    """
    Uses ground truth info only for initial position - position is determined by odometry
    and landmarks by observation therafter. All robots will be in the same global frame.
    """

    def __init__(self, robot_num, gt: pd.DataFrame, landmark_gt: pd.DataFrame, odometry: pd.DataFrame, measurements: pd.DataFrame, barcode_subject_dict) -> None:
        self.timestep = 0
        self.time = 0 # real time
        self.robot_num = robot_num
        self.odometry = odometry
        self.measurements = measurements
        self.barcode_subject_dict = barcode_subject_dict

        self.gt = gt
        self.landmark_gt = landmark_gt


        # self.current_position_v_idx = -1
        self.last_odometry_vertex = None

        # Read the first line from ground truth and initialize the graph and position 
        _, x, y, orientation = gt.iloc[0] # remember gt is a Pandas data frame

        self.pose = PoseSE2([x, y], orientation)


        # Initialize a graph with one vertex and no edges
        self.graph = Graph([], [])
        self.add_new_odometry_vertex()

        self.current_visible_robots = []
        self.other_robot_measurements = {}
        self.seen_landmarks = set()
        self.detected_all_lm_time = (None, float("inf")) # timestep, time

        self.dx_errors = []
        self.dy_errors = []
        self.w_errors = []

        self.total_opt_time = 0.
        
    def update_pose(self, v, w, dt):
        """
        Given existing pose, forward/angular velocity, and timestep duration,
        calculate what the new pose should be. (This will not be
        the true pose as odometry is unreliable.)
        """
        # Compute radius of instantaneous circle of curvature (ICC)
        if w == 0:
            # Going straight ahead without turning
            new_pose = self.pose + PoseSE2([v * dt, 0], 0) # Is this right?
            self.pose = new_pose
            return
        
        r = v / (w * dt)
        x, y, theta = self.pose

        x_new = r*np.cos(w*dt)*np.sin(theta) + r*np.sin(w*dt)*np.cos(theta) + x - r*np.sin(theta)
        y_new = r*np.sin(w*dt)*np.sin(theta) - r*np.cos(theta)*np.cos(w*dt) + y + r*np.cos(theta)
        # x_new = x + np.cos(theta) * v
        # y_new = y + np.sin(theta) * v
        theta_new = theta + w*dt

        self.pose = PoseSE2([x_new, y_new], theta_new)

    def add_new_odometry_vertex(self):
        new_vertex = Vertex(f"{self.robot_num}_{self.timestep}", self.pose)        
        self.graph._vertices.append(new_vertex)

        # Only add edges if this is not the first vertex
        if len(self.graph._vertices) != 1:
            new_odometry_edge = EdgeOdometry(
                vertex_ids=[self.last_odometry_vertex.id, new_vertex.id],
                information=np.linalg.inv(np.diag(GT_ODOMETRY_NOISE_STDEV)), # probably a bad idea
                estimate=(new_vertex.pose - self.last_odometry_vertex.pose),
                vertices=[self.last_odometry_vertex, new_vertex],
            )
            self.graph._edges.append(new_odometry_edge)
        self.last_odometry_vertex = self.graph._vertices[-1]
        self.graph._link_edges()


    def update_current_visible_robots(self, current_obs):
        """
        Get a list of the robots (not landmarks) the robot
        can see at the current timestep. 
        Used by the global sim controller to decide whether
        robots should communicate (if they can't see each other,
        it won't be possible to reach a consensus on relative position.)

        Differs from the GT only robot in that we also record the measurements
        for each visible robot.
        """
        visible_robots = []
        other_robot_measurements = {}
        if current_obs.empty:
            return
        
        for _, obs in current_obs.iterrows():
            # Get the subject# from the barcode number
            try:
                subject_num = self.barcode_subject_dict[obs["Subject"]]
            except Exception:
                print("Ignoring invalid observation:\n", obs)
                continue
            # only consider robots
            if subject_num not in range(1, 6):
                continue


            if subject_num not in visible_robots:
                visible_robots.append(subject_num)
                other_robot_measurements[subject_num] = obs

        self.current_visible_robots = visible_robots
        self.other_robot_measurements = other_robot_measurements

    def rotate_translate_map(self, theta, dx, dy):
        """
        Given an angle 'theta' to rotate by, rotate the map
        around the current point and then translate by 'dx' and 'dy'.
        """
        robot_pose = self.pose.to_array()

        # Rotate around current position
        translation_pose_1 = PoseSE2([robot_pose[0], robot_pose[1]], 0)
        translation_pose_2 = PoseSE2([-robot_pose[0], -robot_pose[1]], 0)
        rotation_pose = PoseSE2([0, 0], theta)

        # translate by (dx, dy)
        translation_pose_3 = PoseSE2([dx, dy], 0)

        for v in self.graph._vertices:
            new_pose_arr = translation_pose_1 + rotation_pose + translation_pose_2 + v.pose.to_array()
            v.pose = translation_pose_3 + PoseSE2([new_pose_arr[0], new_pose_arr[1]], new_pose_arr[2])

        self.graph._link_edges()

        for e in self.graph._edges:
            # Correct the measurements
            e.estimate = (e.estimate - (e.vertices[1].pose - e.vertices[0].pose)).to_compact()

    def consensus_map_frame(self, other_robot):
        measurement_of_other = self.other_robot_measurements[other_robot.robot_num]
        measurement_of_me = other_robot.other_robot_measurements[self.robot_num]

        est_range_to_other = measurement_of_other["Range"]
        bearing_to_other = measurement_of_other["Bearing"]
        
        est_range_to_me = measurement_of_me["Range"]
        bearing_to_me = measurement_of_me["Bearing"]

        other_orientation_in_my_frame = other_robot.pose[2] + bearing_to_me + np.pi - bearing_to_other
        if other_orientation_in_my_frame > np.pi:
            other_orientation_in_my_frame -= 2 * np.pi
        if other_orientation_in_my_frame < -np.pi:
            other_orientation_in_my_frame += 2 * np.pi

        # Other robot's "zero radian" angle expressed in this frame - also the difference between the two frames
        other_zerorad_in_my_frame = other_orientation_in_my_frame - other_robot.pose[2]
        if other_zerorad_in_my_frame > np.pi:
            other_zerorad_in_my_frame -= 2 * np.pi
        if other_zerorad_in_my_frame < -np.pi:
            other_zerorad_in_my_frame += 2 * np.pi
        
        est_relative_rx = np.cos(bearing_to_other) * est_range_to_other
        est_relative_ry = np.sin(bearing_to_other) * est_range_to_other
        est_relative_pose_to_other = PoseSE2([est_relative_rx, est_relative_ry], 0)

        est_other_pose = est_relative_pose_to_other + self.pose 

        other_robot_position_frame_difference = est_other_pose - other_robot.pose

        # Assuming other robot does the same, this will reach global frame consensus between this and other robot        
        self.rotate_translate_map(
            theta=0.5 * other_zerorad_in_my_frame, 
            dx=0.5*other_robot_position_frame_difference[0], 
            dy=0.5*other_robot_position_frame_difference[1],
        )


    def test_map_ops(self):
        v0 = Vertex("0", PoseSE2([-1, -1], 0))
        v1 = Vertex("1", PoseSE2([-1, 1], 0))
        v2 = Vertex("2", PoseSE2([1, 1], 0))
        v3 = Vertex("3", PoseSE2([1, -1], 0))
        e0 = EdgeOdometry(["0", "1"], np.eye(3), v1.pose - v0.pose, [v0, v1])
        e1 = EdgeOdometry(["1", "2"], np.eye(3), v2.pose - v1.pose, [v1, v2])
        e2 = EdgeOdometry(["2", "3"], np.eye(3), v3.pose - v2.pose, [v2, v3])
        e3 = EdgeOdometry(["3", "0"], np.eye(3), v0.pose - v3.pose, [v3, v0])

        g = Graph([e0, e1, e2, e3], [v0, v1, v2, v3])
        g.plot()

        # Rotate by pi/4 about (1, 1)
        self.pose = PoseSE2([1, 1], 0)
        self.graph = g
        self.rotate_translate_map(np.pi / 4, 0, 0)
        g.plot()
        plt.show()
        exit()



    def send_map(self, other_robot):
        """
        Add all vertices and edges from this robot's map not in the other robot's map to the
        other robot's graph. First, a consensus must be reached between the two on their
        relative pose. (This is why they must observe each other to exchange maps.)
        """
        # TODO: Establish consensus on relative pose and reject if the difference is too great.
        # Otherwise, take the average.

        assert other_robot.robot_num in self.current_visible_robots
        assert self.robot_num in other_robot.current_visible_robots

        """
        measurement_of_other = self.other_robot_measurements[other_robot.robot_num]
        measurement_of_me = other_robot.other_robot_measurements[self.robot_num]

        est_range_to_other = measurement_of_other["Range"]
        bearing_to_other = measurement_of_other["Bearing"]
        
        est_range_to_me = measurement_of_me["Range"]
        bearing_to_me = measurement_of_me["Bearing"]

        other_orientation_in_my_frame = other_robot.pose[2] + bearing_to_me + np.pi - bearing_to_other
        if other_orientation_in_my_frame > np.pi:
            other_orientation_in_my_frame -= 2 * np.pi
        if other_orientation_in_my_frame < -np.pi:
            other_orientation_in_my_frame += 2 * np.pi

        

        if (
            abs(est_range_to_other - est_range_to_me) > RELATIVE_RANGE_ERROR_TOLERANCE
        ):
            print("Excessive error in relative robot range; rejecting merge")
            print("\t(Relative ranges: ", est_range_to_other, "/", est_range_to_me, ")")
            # print("\t(Relative bearings: ", bearing_to_other, "/", bearing_to_me, ")")
            return
        
        # Get consensus by averaging the two measurements
        consensus_range = 0.5 * (est_range_to_me + est_range_to_other)

        
        est_relative_rx = np.cos(bearing) * est_range
        est_relative_ry = np.sin(bearing) * est_range
        
        estimated_other_pose = self.pose + 

        graph_copy = copy.deepcopy(self.graph)

        """

        # Add all of "our" vertices not in "their" graph to their graph
        old_other_v_len = len(other_robot.graph._vertices)
        for v in self.graph._vertices:
            if v.id not in [other.id for other in other_robot.graph._vertices]:
                
                other_robot.graph._vertices.append(copy.deepcopy(v))

                # Update the other robot's "seen landmarks" set
                if "_" not in v.id:
                    # Is landmark
                    if v.id not in other_robot.seen_landmarks:
                        # print(v.id, " is not in ", other_robot.seen_landmarks)
                        print("Sharing landmark ", v.id, f" {self.robot_num} --> {other_robot.robot_num}")
                        other_robot.seen_landmarks.add(v.id)

        print("Extended robot ", other_robot.robot_num, f"graph: {old_other_v_len}/{len(other_robot.graph._vertices)} vertices")
        
        old_other_e_len = len(other_robot.graph._vertices)
        for e in self.graph._edges:
            if e not in other_robot.graph._edges:
                modified_edge = copy.deepcopy(e)
                modified_edge.vertices = None
                other_robot.graph._edges.append(modified_edge)

        print("Extended robot ", other_robot.robot_num, f"graph: {old_other_e_len}/{len(other_robot.graph._edges)} edges")

        other_robot.graph._link_edges()

    
    def step(self):
        """
        Advance a single step in the simulation. 
        This involves calculating the robot's new position.
        """
        self.timestep += 1
        old_time = self.time
        old_x = self.pose[0]
        old_y = self.pose[1]
        old_theta = self.pose[2]
        self.time = self.odometry.iloc[self.timestep]["Time"]

        v = self.odometry.iloc[self.timestep]["Velocity"]
        w = self.odometry.iloc[self.timestep]["Angular Velocity"]
        
        dt = self.time - old_time

        self.update_pose(v, w, dt)
        self.add_new_odometry_vertex() # odometry vertex?

        # print("Estimated pose: ", self.pose.to_array())
        _, x, y, orientation = self.gt.iloc[self.timestep] # remember gt is a Pandas data frame

        true_pose = PoseSE2([x, y], orientation)
        self.dx_errors.append(abs((self.pose[0] - old_x) - (x - self.gt.iloc[self.timestep-1][1])))
        self.dy_errors.append(abs((self.pose[1] - old_y) - (y - self.gt.iloc[self.timestep-1][2])))
        self.w_errors.append(abs((self.pose[2] - old_theta) - (orientation - self.gt.iloc[self.timestep-1][3])))


        if self.timestep > 100:
            pass # exit()

        # TODO: Spot any landmarks
        # Get all measurements from the current time (there could be many)
        current_obs = self.measurements.loc[self.measurements["Timestep"] == self.timestep]
        if not current_obs.empty:
            # print("Spotted landmark(s) at timestep ", self.timestep)
            self.update_landmarks(current_obs)
            self.update_current_visible_robots(current_obs)

                        # If all landmarks seen, record time(step)
        if len(self.seen_landmarks) == 15 and self.detected_all_lm_time == (None, float("inf")):
            print(self.robot_num, " detected all landmarks at time ", self.time)
            self.detected_all_lm_time = (self.timestep, self.time)

def load_dataset(args):

    # Lists of Pandas dataframes
    robot_gts = []
    robot_measurements = []
    robot_odometries=[]

    dataset_dir = args.dataset_dir
    barcodes = slam_utils.load_mrclam_file(dataset_dir, "Barcodes.dat")
    landmark_gt = slam_utils.load_mrclam_file(dataset_dir, "Landmark_Groundtruth.dat")


    # Load all robot data
    for i in range(1, args.num_robots + 1): # use data from 1-5 robots
        robot_i_gt = slam_utils.load_mrclam_file(dataset_dir, f"Robot{i}_Groundtruth.dat")
        robot_gts.append(robot_i_gt)

        # Load range/bearing measurements
        robot_i_measurement = slam_utils.load_mrclam_file(dataset_dir, f"Robot{i}_Measurement.dat")
        robot_measurements.append(robot_i_measurement)

        # Load odometry information
        robot_i_odometry = slam_utils.load_mrclam_file(dataset_dir, f"Robot{i}_Odometry.dat")
        robot_odometries.append(robot_i_odometry)

    return barcodes, landmark_gt, robot_gts, robot_measurements, robot_odometries

def compute_landmark_error(landmark_gt, robot: GraphSLAMRobot):
    g: Graph = robot.graph
    landmark_errors = {}
    landmark_estimates = {}
    landmark_gt_coor = {}
    seen_landmarks = set()
    print("------------Evaluating landmark error for robot ", robot.robot_num, "--------------")
    for v in g._vertices:
        if "_" not in v.id:
            # Landmark vertex
            landmark_num = int(v.id) # should be in the range [6, 20]
            if landmark_num in seen_landmarks:
                print("WARNING: Landmark ", v.id, " has more than one vertex in graph!!!")
            seen_landmarks.add(landmark_num)

            l_x, l_y, _ = v.pose.to_array()

            _, gt_x, gt_y, _, _ = landmark_gt.loc[landmark_gt["Subject"] == landmark_num].squeeze()
            
            landmark_errors[landmark_num] = np.sqrt(abs(gt_x - l_x) ** 2  + abs(gt_y - l_y) ** 2)
            landmark_estimates[landmark_num] = (l_x, l_y)
            landmark_gt_coor[landmark_num] = (gt_x, gt_y)

    print("Estimated coordinates: ", landmark_estimates)
    print("True landmark coordinates: ", landmark_gt_coor)
    print("Error per landmark: ", landmark_errors)
    print("Total landmark error: ", sum([error for error in landmark_errors.values()]))
    print("Average landmark error: ", np.mean([error for error in landmark_errors.values()]))

def compute_odometry_error(robot_gts, robot: GraphSLAMRobot):
    g: Graph = robot.graph
    robot_position_sse = 0
    robot_orientation_error = 0
    print("------------Evaluating odometry error for robot ", robot.robot_num, "--------------")
    for v in g._vertices:
        if "_" in v.id:
            # Robot odometry vertex
            vertex_creator_robot, vertex_timestep = list(map(int, v.id.split("_")))
            if vertex_creator_robot != robot.robot_num:
                continue # Only concerned with own odometry error

            # Robots are numbered 1-5
            _, gt_x, gt_y, gt_or = robot_gts[vertex_creator_robot - 1].iloc[vertex_timestep].squeeze()

            robot_position_sse += np.sqrt(abs(gt_x - v.pose[0]) ** 2 + abs(gt_y - v.pose[1]) ** 2)
            robot_orientation_error += abs(gt_or - v.pose[2])

    print("Total pose error: ", f"{robot_position_sse} positional, {robot_orientation_error} orientation")
    print("Average robot pose error: ", f"{robot_position_sse / len(robot_gts[vertex_creator_robot - 1])} positional, {robot_orientation_error/ len(robot_gts[vertex_creator_robot - 1])} orientation")
    print("Average odometry errors: ", np.mean(robot.dx_errors), '/', np.mean(robot.dy_errors), '/', np.mean(robot.w_errors))
def compute_graph_error(args, robot_gts, landmark_gt, robot: GraphSLAMRobot):
    """
    Given a robot's computed SLAM graph, calculate the mean squared error 
    for the robot(s) mapped and every landmark.
    """


    g: Graph = robot.graph
    
    # Robot pose error
    robot_position_sse = np.zeros(args.num_robots)
    robot_orientation_sse = np.zeros(args.num_robots)

    # landmark pose error (disregard orientation)
    landmark_sse = 0

    for v in g._vertices:
        if "_" in v.id:
            # Robot odometry vertex
            vertex_creator_robot, vertex_timestep = list(map(int, v.id.split("_")))

            # Robots are numbered 1-5
            _, gt_x, gt_y, gt_or = robot_gts[vertex_creator_robot - 1].iloc[vertex_timestep].squeeze()
            gt_pose = PoseSE2([gt_x, gt_y], gt_or)

            robot_position_sse += (gt_x - v.pose[0]) ** 2 + (gt_y - v.pose[1]) ** 2 
            robot_orientation_sse += (gt_or - v.pose[2]) ** 2
        else:
            # Landmark vertex
            landmark_num = int(v.id) # should be in the range [6, 20]

            l_x, l_y, _ = v.pose.to_array()

            _, gt_x, gt_y, _, _ = landmark_gt.loc[landmark_gt["Subject"] == landmark_num].squeeze()
            
            landmark_sse += (gt_x - l_x) ** 2 + (gt_y - l_y) ** 2

    # MSE for position/orientation, SSE for landmarks
    # (Note: DOES NOT do existence check on every landmark)
    return np.mean(robot_position_sse), np.mean(robot_orientation_sse), landmark_sse

if __name__ == "__main__":
    # Required Args
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--dataset_dir', required=True, help="Directory of dataset used for this run")
    parser.add_argument('--sample_time', default=0.02, type=float, help="data sampling interval")
    parser.add_argument('--num_robots', default=5, type=int, help="number of robots [1-5]", choices=[1, 2, 3, 4, 5])
    parser.add_argument('--max_timesteps', default=None, type=int, help="Number of timesteps to simulate (if less than dataset length)")
    
    parser.add_argument('--no_share', action="store_true", help="Don't share maps if true")
    
    parser.add_argument('--no_opt', action="store_true", help="Don't optimize maps at the end if true")
    parser.add_argument('--opt_interval', default=None, type=int, help="Optimize maps every x timesteps")
    
    parser.add_argument('--record', action="store_true", help="Export map creation per robot as animated GIF")
    parser.add_argument('--save_interval', default=None, type=int, help="save maps to file every x timesteps")
    
    # Execution mode
    parser.add_argument('--mode', required=True, type=str, choices=["read_only", "gt_only", "with_noise", "with_odo", "test_map"])
    parser.add_argument('--solver', default=None, type=str, choices=["torch", "njit", "numpy", "umfpack", "skcuda", "cusparse", "cholmod"])
    args = parser.parse_args()

    # Load and sample appropriately the required dataset
    barcodes, landmark_gt, robot_gts, robot_measurements, robot_odometries = load_dataset(args)
    timesteps, robot_gts, robot_measurements, robot_odometries = slam_utils.sample_mrclam_dataset(
        args.num_robots, 
        robot_gts, 
        robot_measurements,
        robot_odometries, 
        args.sample_time,
    )

    barcode_mapping = slam_utils.create_barcode_mapping(barcodes)

    print("Landmark gt: ", landmark_gt)

    if args.mode not in ["read_only", "gt_only", "with_noise", "with_odo", "test_map"]:
        raise NotImplementedError(f"Mode {args.mode} is not implemented.")

    if args.mode == "read_only":
        # That's all, folks
        exit()

    if args.mode == "gt_only":
        robots: List[TrivialGTOnlyRobot] = []
        for r in range(args.num_robots):
            robots.append(TrivialGTOnlyRobot(r + 1, robot_gts[r], landmark_gt, robot_measurements[r], barcode_mapping))
    elif args.mode == "with_noise":
        np.random.seed(0) # Make deterministic for debugging/comparison
        robots: List[NoisyGTRobot] = []
        for r in range(args.num_robots):
            robots.append(NoisyGTRobot(r + 1, robot_gts[r], landmark_gt, robot_measurements[r], barcode_mapping))
    elif args.mode == "with_odo":
        np.random.seed(0) # Make deterministic for debugging/comparison
        robots: List[OdometryAndMeasurementRobot] = []
        for r in range(args.num_robots):
            robots.append(OdometryAndMeasurementRobot(r + 1, robot_gts[r], landmark_gt, robot_odometries[r], robot_measurements[r], barcode_mapping))
    elif args.mode == "test_map":
        robots: List[OdometryAndMeasurementRobot] = [OdometryAndMeasurementRobot(1, robot_gts[0], landmark_gt, robot_odometries[0], robot_measurements[0], barcode_mapping)]
        robots[0].test_map_ops()
    if args.max_timesteps is not None:
        assert args.max_timesteps <= timesteps
        timesteps = args.max_timesteps


    # Prepare directory for image save
    if args.record:
        if args.save_interval is None:
            raise ValueError("You must specify a save interval when recording the results")
        for r in range(args.num_robots):
            d = Path(args.dataset_dir + '/saved_map_' + str(r + 1))
            d.mkdir(parents=True,exist_ok=True)

    controller = GlobalSimController(robots, robot_gts, timesteps, args)
    print("BEGINNING SIMULATION: running for ", timesteps, " timesteps")

    controller.execute()

    # Evaluate at end
    for robot in robots:
        print("Robot ", robot.robot_num, " seen landmarks: ", len(robot.seen_landmarks))
        if len(robot.seen_landmarks) == 15:
            print("Robot ", robot.robot_num, " detected all landmarks at timestep ", f"{robot.detected_all_lm_time[0]} ({robot.detected_all_lm_time[1]}s)")
        else:
            print("Robot ", robot.robot_num, f" failed to locate all landmarks! (found {len(robot.seen_landmarks)}/15)")
            print(robot.seen_landmarks)

        compute_landmark_error(landmark_gt, robot)
        compute_odometry_error(robot_gts, robot)
        
        robot.graph.plot(title=f"Map from robot {robot.robot_num}")
    plt.show()

