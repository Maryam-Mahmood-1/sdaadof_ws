import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider  # Import Slider Widget

# IMPORT MODULES
from some_examples_py.robot_dynamics import RobotDynamics
from some_examples_py.resclf_formulation import RESCLF_Formulation
from some_examples_py.qp_solver import solve_qp

# --- CONFIGURATION ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class RESCLF_Controller_Node(Node):
    def __init__(self):
        super().__init__('resclf_modular_node')
        
        # Initialize Theory Modules
        self.robot = RobotDynamics(URDF_PATH, 'endeffector', TARGET_JOINTS)
        self.resclf = RESCLF_Formulation(dim=3)
        
        # Trajectory Parameters
        self.center_pos = np.array([0.0, 0.0, 0.72])
        self.ellipse_a = 0.15 
        self.ellipse_b = 0.27 
        self.period = 7.5
        
        # ROS Communication
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(0.002, self.control_loop)
        
        self.start_time = None
        self.received_state = False
        self.start_approach_pos = None  
        
        # Data Logging (We keep accumulating everything here)
        self.log_target = {'x':[], 'y':[]}
        self.log_actual = {'x':[], 'y':[]}

    def cb_joints(self, msg):
        msg_map = {name: i for i, name in enumerate(msg.name)}
        self.robot.update_state_from_ros(msg, msg_map, TARGET_JOINTS)
        self.received_state = True

    def get_trajectory(self, t):
        omega = 2 * math.pi / self.period
        angle = omega * t
        
        pd = self.center_pos + np.array([
            self.ellipse_a * math.cos(angle), 
            self.ellipse_b * math.sin(angle), 
            0.0
        ])
        vd = np.array([
            -self.ellipse_a * omega * math.sin(angle), 
             self.ellipse_b * omega * math.cos(angle), 
             0.0
        ])
        ad = np.array([
            -self.ellipse_a * (omega**2) * math.cos(angle), 
            -self.ellipse_b * (omega**2) * math.sin(angle), 
             0.0
        ])
        return pd, vd, ad

    def control_loop(self):
        if not self.received_state: return
        
        M, nle, p, v, J, dJ_dq = self.robot.compute()
        
        if self.start_time is None: 
            self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        t = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        
        if t < 5.0: 
            start_pt = self.center_pos + np.array([self.ellipse_a, 0, 0])
            if self.start_approach_pos is None:
                self.start_approach_pos = p
            ratio = t / 5.0
            sm_ratio = (1 - math.cos(ratio * math.pi)) / 2
            pd = (1 - sm_ratio) * self.start_approach_pos + sm_ratio * start_pt
            vd, ad = np.zeros(3), np.zeros(3)
        else:
            pd, vd, ad = self.get_trajectory(t - 5.0)

        # Log Data
        self.log_actual['x'].append(p[0])
        self.log_actual['y'].append(p[1])
        self.log_target['x'].append(pd[0])
        self.log_target['y'].append(pd[1])

        e = p - pd
        de = v - vd
        
        LfV, LgV, V_val, gamma = self.resclf.get_qp_constraints(e, de)
        mu = solve_qp(LfV, LgV, V_val, gamma, e, de)
        
        J_pinv = np.linalg.pinv(J)
        x_ddot_command = ad + mu
        q_ddot_command = J_pinv @ (x_ddot_command - dJ_dq)
        tau = (M @ q_ddot_command) + nle
        
        tau = np.clip(tau[self.robot.v_indices], -50.0, 50.0)
        
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub.publish(msg)

    def stop(self):
        msg = Float64MultiArray(data=[0.0]*7)
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RESCLF_Controller_Node()
    
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    
    # --- PLOTTING SETUP ---
    fig, ax = plt.subplots(figsize=(8, 10))
    # Adjust layout to leave room at the bottom for the slider
    plt.subplots_adjust(bottom=0.15) 
    
    ln_t, = ax.plot([],[], 'b--', label='Target')
    ln_a, = ax.plot([],[], 'r-', label='Actual')
    
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4) 
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel('Robot Y [m]') 
    ax.set_ylabel('Robot X [m]')  
    ax.set_title('Operational Space Trajectory')
    ax.legend()
    ax.grid(True)

    # --- HISTORY SLIDER SETUP ---
    # Define axes for slider [left, bottom, width, height]
    ax_hist = plt.axes([0.2, 0.05, 0.65, 0.03])
    
    # Range: 10 points (min) to 5000 points (max)
    # Valinit: Start with 500 points visible
    slider_history = Slider(
        ax_hist, 'History Points', 10, 7500, valinit=500, valstep=10
    )

    def update(frame):
        # 1. Grab references to the lists
        ty = node.log_target['y']
        tx = node.log_target['x']
        ay = node.log_actual['y']
        ax_data = node.log_actual['x']
        
        # 2. Get safe max length
        min_t = min(len(ty), len(tx))
        min_a = min(len(ay), len(ax_data))
        
        # 3. Get history length from slider
        hist_len = int(slider_history.val)
        
        # 4. Determine start index (slicing)
        # If we have 1000 points and want to show 200, start at 800.
        # If we have 100 points and want to show 200, start at 0.
        start_t = max(0, min_t - hist_len)
        start_a = max(0, min_a - hist_len)
        
        # 5. Plot the slice
        ln_t.set_data(ty[start_t:min_t], tx[start_t:min_t])
        ln_a.set_data(ay[start_a:min_a], ax_data[start_a:min_a])
        
        return ln_t, ln_a
        
    ani = FuncAnimation(fig, update, interval=100)
    plt.show()
    
    node.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()