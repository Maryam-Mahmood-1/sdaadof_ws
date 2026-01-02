import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.patches import Rectangle

# --- IMPORT NEW ROBUST MODULES ---
# 1. Robot Dynamics (Shared, unchanged)
from some_examples_py.CLF_CBF_traj.robot_dynamics import RobotDynamics

# 2. CR-CLF (Stability with Robustness Term)
from some_examples_py.CRCLF_CRCBF_traj.crclf_formulation import CRCLF_Formulation

# 3. CR-CBF (Safety with Robustness Term)
from some_examples_py.CRCLF_CRCBF_traj.crcbf_formulation import CRCBF_Formulation

# 4. CR-QP Solver (Accepts robust_term)
from some_examples_py.CRCLF_CRCBF_traj.crqp_solver import solve_qp

# --- CONFIGURATION ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy.urdf"
TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class CR_Controller_Node(Node):
    def __init__(self):
        super().__init__('cr_modular_node')
        
        # --- 1. CONFORMAL QUANTILE ---
        # PASTE YOUR CALCULATED VALUE HERE!
        # This is the "Safety Margin" derived from your calibration data.
        self.q_cp = 0.29861  # <--- REPLACE THIS WITH YOUR VALUE FROM compute_quantile.py

        print(f"Initializing CR-Controller with Quantile (q_cp): {self.q_cp}")

        # --- 2. Initialize Dynamics ---
        self.robot = RobotDynamics(URDF_PATH, 'endeffector', TARGET_JOINTS)
        
        # [cite_start]--- 3. Initialize CR-CLF (Stability) [cite: 50] ---
        # We pass q_cp so it can calculate ||LgV|| * q
        self.crclf = CRCLF_Formulation(dim=3, q_cp=self.q_cp)
        
        # [cite_start]--- 4. Initialize CR-CBF (Safety) [cite: 50, 168] ---
        # We pass q_cp so it can calculate ||grad_h|| * q and tighten the constraint
        self.crcbf = CRCBF_Formulation(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.2, 0.2, 0.4], 
            power_n=4,
            q_cp=self.q_cp
        )
        
        self.cbf_active = False 

        # Trajectory Parameters
        self.center_pos = np.array([0.0, 0.0, 0.72])
        self.ellipse_a = 0.15 
        self.ellipse_b = 0.27 
        self.period = 12.0
        
        # ROS Communication
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(0.002, self.control_loop)
        
        self.start_time = None
        self.received_state = False
        self.start_approach_pos = None  
        
        # Data Logging
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
        
        # 1. Dynamics
        M, nle, p, v, J, dJ_dq = self.robot.compute()
        
        # 2. Trajectory Generation
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

        # 3. Formulate Errors
        e = p - pd
        de = v - vd
        
        # --- 4. Get Stability Constraints (CR-CLF) ---
        #[cite_start]# [cite: 145] "Robustify the standard CLF using quantified uncertainty"
        # Returns robust_term which accounts for q_cp
        LfV, LgV, V_val, gamma, robust_term = self.crclf.get_qp_constraints(e, de)
        
        # --- 5. Get Safety Constraints (CR-CBF) ---
        cbf_L, cbf_b = None, None
        
        if self.cbf_active:
            #[cite_start]# [cite: 170] Returns constraints where b is already tightened by q_cp
            cbf_L, cbf_b = self.crcbf.get_constraints(p, v, ad)
        
        # --- 6. Solve Unified QP ---
        # We pass robust_term so the solver can tighten the CLF bound
        mu = solve_qp(LfV, LgV, V_val, gamma, e, de, cbf_L, cbf_b, robust_term)
        
        # 7. Compute Torque
        J_pinv = np.linalg.pinv(J)
        x_ddot_command = ad + mu
        q_ddot_command = J_pinv @ (x_ddot_command - dJ_dq)
        tau = (M @ q_ddot_command) + nle
        
        # Safety Clip (using URDF limits ideally, or hardcoded for now)
        tau = np.clip(tau[self.robot.v_indices], -45.0, 45.0)
        
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub.publish(msg)

    def stop(self):
        msg = Float64MultiArray(data=[0.0]*7)
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CR_Controller_Node()
    
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    
    # --- PLOTTING SETUP ---
    fig, ax = plt.subplots(figsize=(8, 10))
    plt.subplots_adjust(bottom=0.20) 
    
    ln_t, = ax.plot([],[], 'b--', label='Target')
    ln_a, = ax.plot([],[], 'r-', label='Actual (Robust)')
    
    # Draw Safe Set (Using dimensions from CR-CBF)
    rx = node.crcbf.dims[0]
    ry = node.crcbf.dims[1]
    
    anchor_x = -ry
    anchor_y = -rx
    width = 2 * ry
    height = 2 * rx
    
    safe_rect = Rectangle(
        (anchor_x, anchor_y), width, height, 
        linewidth=2, edgecolor='g', facecolor='none', 
        linestyle='-', label='Safe Set (CR-CBF)'
    )
    ax.add_patch(safe_rect)
    
    limit_pad = 0.1
    ax.set_xlim(-(ry + limit_pad), (ry + limit_pad))
    ax.set_ylim(-(rx + limit_pad), (rx + limit_pad)) 
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Robot Y [m]') 
    ax.set_ylabel('Robot X [m]')  
    ax.set_title('Conformal Robust Controller (CR-CLF + CR-CBF)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # --- WIDGETS ---
    ax_hist = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider_history = Slider(ax_hist, 'Tail Length', 10, 5000, valinit=500, valstep=10)
    
    ax_check = plt.axes([0.05, 0.05, 0.15, 0.1])
    check = CheckButtons(ax_check, ['Enable\nCR-CBF'], [False])
    
    def toggle_cbf(label):
        node.cbf_active = not node.cbf_active
        print(f"Safety CR-CBF Active: {node.cbf_active}")
        
    check.on_clicked(toggle_cbf)

    def update(frame):
        ty = node.log_target['y']
        tx = node.log_target['x']
        ay = node.log_actual['y']
        ax_data = node.log_actual['x']
        
        min_t = min(len(ty), len(tx))
        min_a = min(len(ay), len(ax_data))
        
        hist_len = int(slider_history.val)
        start_t = max(0, min_t - hist_len)
        start_a = max(0, min_a - hist_len)
        
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