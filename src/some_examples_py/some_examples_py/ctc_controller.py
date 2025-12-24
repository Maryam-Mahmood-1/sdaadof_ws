import sys
import os

# --- 1. CONDA + ROS FIX ---
# (Essential if running from the 'pinocchio_robotics' environment)
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import numpy as np
import time
import math

# --- 2. USER CONFIGURATION ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_2.urdf"
COMMAND_TOPIC = '/effort_arm_controller/commands' 

# IMPORTANT: List joints EXACTLY as they appear in your config/my_controllers.yaml
# If this order is wrong, the wrong motor will get the wrong torque!
CONTROLLER_JOINT_ORDER = [
    'joint_1', 
    'joint_2', 
    'joint_3', 
    'joint_4', 
    'joint_5', 
    'joint_6', 
    'joint_7'
]

class PinocchioGazeboController(Node):
    def __init__(self):
        super().__init__('pinocchio_ctc_controller')
        
        # Setup Pinocchio
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # Internal state
        self.q = np.zeros(self.nq)
        self.v = np.zeros(self.nv)
        self.state_received = False

        # Pre-calculate mapping from Controller names to Pinocchio IDs
        self.output_map = []
        for name in CONTROLLER_JOINT_ORDER:
            if self.model.existJointName(name):
                # Pinocchio ID - 1 = q index
                idx = self.model.getJointId(name) - 1
                self.output_map.append(idx)
            else:
                self.get_logger().error(f"Joint '{name}' not found in URDF!")

        # ROS Interface
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        
        # Control Loop (1 kHz)
        self.dt = 0.001
        self.create_timer(self.dt, self.control_loop)
        
        self.start_time = time.time()
        self.get_logger().info(f"CTC Controller Ready. Waiting for Joint States...")
        
        # Gains
        self.kp = 100.0 
        self.kd = 2 * np.sqrt(self.kp)

    def joint_state_callback(self, msg):
        """Read the 'Real' robot state from Gazebo"""
        if not self.state_received:
            self.get_logger().info("Received first state from Gazebo!")
            self.state_received = True

        for i, name in enumerate(msg.name):
            if self.model.existJointName(name):
                j_id = self.model.getJointId(name)
                idx = j_id - 1
                if 0 <= idx < self.nq:
                    self.q[idx] = msg.position[i]
                    self.v[idx] = msg.velocity[i]

    def control_loop(self):
        if not self.state_received:
            return

        t = time.time() - self.start_time
        
        # --- A. SINE WAVE TRAJECTORY ---
        q_des = np.zeros(self.nq)
        v_des = np.zeros(self.nv)
        a_des = np.zeros(self.nv)
        
        amp = 0.5
        freq = 1.0
        
        for i in range(self.nq):
            q_des[i] = amp * math.sin(freq * t + (i * 0.5))
            v_des[i] = amp * freq * math.cos(freq * t + (i * 0.5))
            a_des[i] = -amp * (freq**2) * math.sin(freq * t + (i * 0.5))

        # --- B. COMPUTED TORQUE CONTROL ---
        error_q = pin.difference(self.model, self.q, q_des)
        error_v = v_des - self.v
        
        u = a_des + self.kp * error_q + self.kd * error_v
        tau = pin.rnea(self.model, self.data, self.q, self.v, u)
        
        # --- C. REORDER & SEND TO GAZEBO ---
        # Map the Pinocchio torque vector (tau) to the Controller's expected order
        ordered_tau = [tau[idx] for idx in self.output_map]
        
        msg = Float64MultiArray()
        msg.data = ordered_tau
        self.torque_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PinocchioGazeboController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()