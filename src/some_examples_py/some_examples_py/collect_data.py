#!/usr/bin/env python3
import sys
import os
import time
import math
import numpy as np
import csv
import datetime

# --- 1. CONDA + ROS FIX ---
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin

# --- 2. CONFIGURATION ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_2.urdf"
COMMAND_TOPIC = '/effort_arm_controller/commands' 

# Output Path
SAVE_DIR = "/home/maryammahmood/xdaadbot_ws/"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

DATA_FILE = f"{SAVE_DIR}robot_data_centered_{TIMESTAMP}.csv"

# --- 3. FILTER SETTINGS ---
CONTROLLER_JOINT_ORDER = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

# Parameters
TOTAL_EPISODES = 100
EPISODE_DURATION = 5.0 # seconds
DT = 0.001 

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')
        
        # A. Setup Pinocchio
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # B. Internal Robot State
        self.q_current = np.zeros(self.nq)
        self.v_current = np.zeros(self.nv)
        self.v_prev = np.zeros(self.nv) 
        self.state_received = False
        
        # Alignment Buffers
        self.prev_q = None
        self.prev_v = None
        self.prev_tau = None
        
        # C. Create Filter Mappings
        self.save_indices = []
        
        self.get_logger().info("Mapping joints & checking limits...")
        for name in CONTROLLER_JOINT_ORDER:
            if self.model.existJointName(name):
                idx = self.model.getJointId(name) - 1
                self.save_indices.append(idx)
                
                # Print limits for verification
                lim_low = self.model.lowerPositionLimit[idx]
                lim_high = self.model.upperPositionLimit[idx]
                self.get_logger().info(f"  {name}: Index {idx} | Limits [{lim_low:.2f}, {lim_high:.2f}]")
            else:
                self.get_logger().error(f"  CRITICAL: Joint '{name}' not found!")

        self.save_indices = np.array(self.save_indices, dtype=int)

        # D. ROS Interface
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        
        # E. REAL-TIME FILE SETUP
        self.get_logger().info(f"Opening file: {DATA_FILE}")
        self.csv_file = open(DATA_FILE, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header
        header = ["episode_id"] + \
                 [f"q_{name}" for name in CONTROLLER_JOINT_ORDER] + \
                 [f"v_{name}" for name in CONTROLLER_JOINT_ORDER] + \
                 [f"tau_{name}" for name in CONTROLLER_JOINT_ORDER] + \
                 [f"acc_{name}" for name in CONTROLLER_JOINT_ORDER]
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        
        # F. Control Loop
        self.timer = self.create_timer(DT, self.control_loop)
        
        # Gains
        self.kp = 100.0 
        self.kd = 2 * np.sqrt(self.kp)

        # G. Episode Management
        self.current_episode = 0
        self.episode_start_time = 0.0
        self.is_collecting = False
        
        # Initialize trajectory arrays
        self.q_center = np.zeros(self.nq)
        self.amp = np.zeros(self.nq)
        self.phase_offset = np.zeros(self.nq)
        
        self.reset_trajectory_params()
        
    def reset_trajectory_params(self):
        """Generates random Centers and Amplitudes within valid limits"""
        self.omega = np.random.uniform(0.5, 2.0)
        
        for i in range(self.nq):
            # 1. Get Limits for this specific joint
            lim_low = self.model.lowerPositionLimit[i]
            lim_high = self.model.upperPositionLimit[i]
            
            # Handle infinite/unbounded joints (set arbitrary safe range)
            if lim_high > 100 or lim_low < -100:
                lim_low, lim_high = -np.pi, np.pi

            # 2. Pick a random Amplitude first (0.1 to 0.5 rad)
            a = np.random.uniform(0.1, 0.5)
            
            # 3. Determine the "Safe Zone" for the center
            # The center must be far enough from edges so (center + amp) < high
            safe_min = lim_low + a
            safe_max = lim_high - a
            
            # 4. Pick Center
            if safe_min >= safe_max:
                # Range is too tight? Center strictly in middle, reduce amp
                c = (lim_low + lim_high) / 2.0
                a = (lim_high - lim_low) / 2.0 * 0.9 # 90% of range
            else:
                # Pick random center in safe zone
                c = np.random.uniform(safe_min, safe_max)
            
            self.q_center[i] = c
            self.amp[i] = a
            self.phase_offset[i] = np.random.uniform(0, math.pi)

    def joint_state_callback(self, msg):
        if not self.state_received:
            self.get_logger().info("Connected to Gazebo!")
            self.state_received = True
            self.start_new_episode()

        for i, name in enumerate(msg.name):
            if self.model.existJointName(name):
                j_id = self.model.getJointId(name)
                idx = j_id - 1
                if 0 <= idx < self.nq:
                    self.q_current[idx] = msg.position[i]
                    self.v_current[idx] = msg.velocity[i]

    def start_new_episode(self):
        if self.current_episode >= TOTAL_EPISODES:
            self.close_file()
            self.get_logger().info("Collection Complete. File closed.")
            raise SystemExit
            
        self.current_episode += 1
        self.reset_trajectory_params()
        
        # Reset alignment buffers
        self.prev_q = None 
        self.prev_v = None
        self.prev_tau = None
        self.v_prev = self.v_current.copy()
        
        self.episode_start_time = time.time()
        self.is_collecting = True
        self.get_logger().info(f"Start Ep {self.current_episode}/{TOTAL_EPISODES} | Center J1: {self.q_center[0]:.2f}")

    def control_loop(self):
        if not self.state_received or not self.is_collecting:
            return

        now = time.time()
        t = now - self.episode_start_time
        
        if t > EPISODE_DURATION:
            self.is_collecting = False
            self.start_new_episode()
            return

        # --- 1. OBSERVE RESULT ---
        accel_full = (self.v_current - self.v_prev) / DT
        
        # --- 2. WRITE FILTERED DATA ---
        if self.prev_q is not None:
            # Extract only the 7 relevant joints
            q_saved   = self.prev_q[self.save_indices]
            v_saved   = self.prev_v[self.save_indices]
            tau_saved = self.prev_tau[self.save_indices]
            acc_saved = accel_full[self.save_indices]
            
            row = np.concatenate([
                [self.current_episode], 
                q_saved, v_saved, tau_saved, acc_saved
            ])
            self.csv_writer.writerow(row)

        self.v_prev = self.v_current.copy()

        # --- 3. GENERATE TRAJECTORY (With Random Centers) ---
        q_des = np.zeros(self.nq)
        v_des = np.zeros(self.nv)
        a_des = np.zeros(self.nv)
        
        for i in range(self.nq):
            p = self.phase_offset[i]
            c = self.q_center[i] # Random center for this episode
            a = self.amp[i]      # Random amp for this episode
            
            # Position: Center + Sine
            q_des[i] = c + a * math.sin(self.omega * t + p)
            
            # Velocity: Derivative of Sine (Center is constant -> 0)
            v_des[i] = a * self.omega * math.cos(self.omega * t + p)
            
            # Acceleration
            a_des[i] = -a * (self.omega**2) * math.sin(self.omega * t + p)

        # --- 4. COMPUTE TORQUE ---
        error_q = pin.difference(self.model, self.q_current, q_des)
        error_v = v_des - self.v_current
        u = a_des + self.kp * error_q + self.kd * error_v
        
        tau_current = pin.rnea(self.model, self.data, self.q_current, self.v_current, u)
        
        # --- 5. STORE & PUBLISH ---
        self.prev_q = self.q_current.copy()
        self.prev_v = self.v_current.copy()
        self.prev_tau = tau_current.copy()

        ordered_tau = tau_current[self.save_indices]
        msg = Float64MultiArray()
        msg.data = ordered_tau.tolist()
        self.torque_pub.publish(msg)

    def close_file(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info("CSV File Closed Safely.")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("Stopping...")
    finally:
        node.close_file()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()


    # #!/usr/bin/env python3
# import sys
# import os
# import time
# import math
# import numpy as np
# import csv
# import datetime

# # --- 1. CONDA + ROS FIX ---
# sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
# sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import pinocchio as pin

# # --- 2. CONFIGURATION ---
# URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_2.urdf"
# COMMAND_TOPIC = '/effort_arm_controller/commands' 

# # Output Path
# SAVE_DIR = "/home/maryammahmood/xdaadbot_ws/"
# TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# DATA_FILE = f"{SAVE_DIR}robot_data_filtered_{TIMESTAMP}.csv"

# # --- 3. FILTER SETTINGS ---
# # ONLY these joints will be saved to the CSV.
# CONTROLLER_JOINT_ORDER = [
#     'joint_1', 'joint_2', 'joint_3', 'joint_4', 
#     'joint_5', 'joint_6', 'joint_7'
# ]

# # Parameters
# TOTAL_EPISODES = 100
# EPISODE_DURATION = 5.0 # seconds
# DT = 0.001 

# class DataCollectorNode(Node):
#     def __init__(self):
#         super().__init__('data_collector_node')
        
#         # A. Setup Pinocchio
#         self.model = pin.buildModelFromUrdf(URDF_PATH)
#         self.data = self.model.createData()
#         self.nq = self.model.nq
#         self.nv = self.model.nv
        
#         # B. Internal Robot State (Full Size)
#         self.q_current = np.zeros(self.nq)
#         self.v_current = np.zeros(self.nv)
#         self.v_prev = np.zeros(self.nv) 
#         self.state_received = False
        
#         # Alignment Buffers
#         self.prev_q = None
#         self.prev_v = None
#         self.prev_tau = None
        
#         # C. Create Filter Mappings
#         self.save_indices = []
        
#         self.get_logger().info("Mapping joints for filtering...")
#         for name in CONTROLLER_JOINT_ORDER:
#             if self.model.existJointName(name):
#                 # Pinocchio joint IDs start at 1 (universe=0). 
#                 # For standard revolute joints, the q/v index is (id - 1).
#                 idx = self.model.getJointId(name) - 1
#                 self.save_indices.append(idx)
#                 self.get_logger().info(f"  Mapped '{name}' to Index {idx}")
#             else:
#                 self.get_logger().error(f"  CRITICAL: Joint '{name}' not found in URDF!")

#         # Convert to numpy array for easy indexing later
#         self.save_indices = np.array(self.save_indices, dtype=int)

#         # D. ROS Interface
#         self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
#         self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        
#         # E. REAL-TIME FILE SETUP
#         self.get_logger().info(f"Opening file: {DATA_FILE}")
#         self.csv_file = open(DATA_FILE, 'w', newline='')
#         self.csv_writer = csv.writer(self.csv_file)
        
#         # Dynamic Header based on joint names
#         header = ["episode_id"] + \
#                  [f"q_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"v_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"tau_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"acc_{name}" for name in CONTROLLER_JOINT_ORDER]
#         self.csv_writer.writerow(header)
#         self.csv_file.flush()
        
#         # F. Control Loop
#         self.timer = self.create_timer(DT, self.control_loop)
        
#         # Gains
#         self.kp = 100.0 
#         self.kd = 2 * np.sqrt(self.kp)

#         # G. Episode Management
#         self.current_episode = 0
#         self.episode_start_time = 0.0
#         self.is_collecting = False
#         self.reset_trajectory_params()
        
#     def reset_trajectory_params(self):
#         self.omega = np.random.uniform(0.5, 2.0)
#         self.amp = np.random.uniform(0.1, 0.6)
#         # Random phase for ALL joints (full model), we just control the active ones
#         self.phase_offset = np.random.uniform(0, math.pi, self.nq)

#     def joint_state_callback(self, msg):
#         if not self.state_received:
#             self.get_logger().info("Connected to Gazebo!")
#             self.state_received = True
#             self.start_new_episode()

#         for i, name in enumerate(msg.name):
#             if self.model.existJointName(name):
#                 j_id = self.model.getJointId(name)
#                 idx = j_id - 1
#                 if 0 <= idx < self.nq:
#                     self.q_current[idx] = msg.position[i]
#                     self.v_current[idx] = msg.velocity[i]

#     def start_new_episode(self):
#         if self.current_episode >= TOTAL_EPISODES:
#             self.close_file()
#             self.get_logger().info("Collection Complete. File closed.")
#             raise SystemExit
            
#         self.current_episode += 1
#         self.reset_trajectory_params()
        
#         # Reset buffers
#         self.prev_q = None 
#         self.prev_v = None
#         self.prev_tau = None
#         self.v_prev = self.v_current.copy()
        
#         self.episode_start_time = time.time()
#         self.is_collecting = True
#         self.get_logger().info(f"Starting Episode {self.current_episode}/{TOTAL_EPISODES}")

#     def control_loop(self):
#         if not self.state_received or not self.is_collecting:
#             return

#         now = time.time()
#         t = now - self.episode_start_time
        
#         if t > EPISODE_DURATION:
#             self.is_collecting = False
#             self.start_new_episode()
#             return

#         # --- 1. OBSERVE RESULT (Full State Accel) ---
#         accel_full = (self.v_current - self.v_prev) / DT
        
#         # --- 2. WRITE FILTERED DATA TO DISK ---
#         if self.prev_q is not None:
#             # EXTRACT ONLY THE 7 JOINTS
#             q_saved   = self.prev_q[self.save_indices]
#             v_saved   = self.prev_v[self.save_indices]
#             tau_saved = self.prev_tau[self.save_indices]
#             acc_saved = accel_full[self.save_indices]
            
#             # Combine into one row
#             row = np.concatenate([
#                 [self.current_episode], 
#                 q_saved, 
#                 v_saved, 
#                 tau_saved, 
#                 acc_saved
#             ])
#             self.csv_writer.writerow(row)

#         self.v_prev = self.v_current.copy()

#         # --- 3. GENERATE TRAJECTORY ---
#         q_des = np.zeros(self.nq)
#         v_des = np.zeros(self.nv)
#         a_des = np.zeros(self.nv)
        
#         # Only generating excitation for the joints we care about? 
#         # Actually it's safer to generate for all, and just use the ones we need.
#         for i in range(self.nq):
#             p = self.phase_offset[i]
#             q_des[i] = self.amp * math.sin(self.omega * t + p)
#             v_des[i] = self.amp * self.omega * math.cos(self.omega * t + p)
#             a_des[i] = -self.amp * (self.omega**2) * math.sin(self.omega * t + p)

#         # --- 4. COMPUTE TORQUE (Full Model) ---
#         error_q = pin.difference(self.model, self.q_current, q_des)
#         error_v = v_des - self.v_current
#         u = a_des + self.kp * error_q + self.kd * error_v
        
#         tau_current = pin.rnea(self.model, self.data, self.q_current, self.v_current, u)
        
#         # --- 5. STORE & PUBLISH ---
#         self.prev_q = self.q_current.copy()
#         self.prev_v = self.v_current.copy()
#         self.prev_tau = tau_current.copy()

#         # Send filtered torque to Gazebo (using the same indices, convenient!)
#         ordered_tau = tau_current[self.save_indices]
#         msg = Float64MultiArray()
#         msg.data = ordered_tau.tolist()
#         self.torque_pub.publish(msg)

#     def close_file(self):
#         if hasattr(self, 'csv_file') and not self.csv_file.closed:
#             self.csv_file.flush()
#             self.csv_file.close()
#             self.get_logger().info("CSV File Closed Safely.")

# def main(args=None):
#     rclpy.init(args=args)
#     node = DataCollectorNode()
#     try:
#         rclpy.spin(node)
#     except (KeyboardInterrupt, SystemExit):
#         node.get_logger().info("Stopping...")
#     finally:
#         node.close_file()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()