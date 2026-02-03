"""
Main Node for 2-DOF Robot using GAZEBO Simulation
CONTROLLER: Learned Model (Data-Driven Inverse Dynamics)
FEATURE: Toggle for Robustness + Strict Initialization + Zero-Start Nudge
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import pickle
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 

# --- CONFIGURATIONS ---
from ament_index_python.packages import get_package_share_directory

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.robot_dynamics import RobotDynamics
from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
from some_examples_py.CRCLF_CRCBF_2_link.resclf_controller import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

URDF_CTRL = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf", "2_link_urdf", "2link_robot.urdf.xacro" 
)

MODEL_PATH = os.path.join(os.path.expanduser("~"), "xdaadbot_ws", "my_learned_robot2.pkl")
EE_NAMES = ["endEffector"]
ALL_JOINTS = ["baseHinge", "interArm"]

class LearnedModelWrapper:
    def __init__(self, model_path):
        print(f"[LearnedModel] Loading from: {model_path}")
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.safety_bounds = data["safety_bounds"]
            print(f"[LearnedModel] Success. Safety Bounds Loaded.")
        except Exception as e:
            print(f"[Error] Could not load model: {e}")
            sys.exit(1)

    def build_features(self, q, dq, tau):
        q1, q2 = q[0], q[1]
        dq1, dq2 = dq[0], dq[1]
        t1, t2 = tau[0], tau[1]
        s1, c1 = np.sin(q1), np.cos(q1)
        s2, c2 = np.sin(q2), np.cos(q2)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        return np.array([[t1, t2, dq1, dq2, s1, c1, s2, c2, s12, c12, dq1**2, dq2**2]])

    def get_inverse_dynamics(self, q, dq):
        feat_0 = self.build_features(q, dq, np.zeros(2))
        b = self.model.predict(feat_0)[0]
        feat_1 = self.build_features(q, dq, np.array([1.0, 0.0]))
        col_1 = self.model.predict(feat_1)[0] - b
        feat_2 = self.build_features(q, dq, np.array([0.0, 1.0]))
        col_2 = self.model.predict(feat_2)[0] - b
        A = np.column_stack([col_1, col_2])
        return np.linalg.pinv(A), b

class GazeboResclfNode(Node):
    def __init__(self):
        super().__init__('gazebo_learned_ctrl_node')
        
        self.sensing_model = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.learned_model = LearnedModelWrapper(MODEL_PATH)
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.2, 1.2, 3.0], power_n=4)
        
        self.cbf_active = False 
        self.use_quantile = False 
        self.learned_quantile_val = self.learned_model.safety_bounds['x']

        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.timer = self.create_timer(1.0/100.0, self.control_loop) 
        
        self.start_time = None
        self.q = None  
        self.dq = np.zeros(2)
        self.tau_limits = np.array([50.0, 30.0])

        self.lock = threading.Lock()
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[], 'V':[], 'err':[]}

    def cb_joints(self, msg):
        try:
            q_buf = [0.0] * 2; dq_buf = [0.0] * 2; found = 0
            for i, name in enumerate(ALL_JOINTS):
                if name in msg.name:
                    idx = msg.name.index(name)
                    q_buf[i] = msg.position[idx]; dq_buf[i] = msg.velocity[idx]; found += 1
            if found == 2: 
                with self.lock:
                    self.q = np.array(q_buf); self.dq = np.array(dq_buf)
        except: pass

    def control_loop(self):
        # 1. INITIALIZATION GATE
        if self.q is None: return

        if self.start_time is None: 
            self.start_time = time.time()
            print(f"[Sync] Initial State Captured. Trajectory Starting.")
        
        t_clock = time.time() - self.start_time

        # 2. SENSING
        with self.lock:
            curr_q, curr_dq = self.q.copy(), self.dq.copy()
            
        _, _, J, dJ, x, dx = self.sensing_model.compute_dynamics(curr_q, curr_dq)
        x_2d, dx_2d = x[0:2], dx[0:2]

        # 3. TRAJECTORY GENERATION
        xd_full, vd_full, ad_full = self.traj_gen.get_ref(
            t_clock, 
            current_actual_pos=np.pad(x_2d, (0,1)),
            current_actual_vel=np.pad(dx_2d, (0,1))
        )
        xd, vd, ad = xd_full[:2], vd_full[:2], ad_full[:2]

        # 4. NOMINAL CONTROL (CLF)
        # SILENCE Feedback and Feedforward for first 0.1s to prevent the "jump"
        if t_clock < 0.1:
            u_ref = np.zeros(2)
        else:
            u_nominal = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
            u_ref = ad + u_nominal
        
        current_q_val = self.learned_quantile_val if self.use_quantile else 0.0
        LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
            x_2d, dx_2d, xd, vd, q_quantile=current_q_val
        )

        # 5. CBF SAFETY
        cbf_A, cbf_b = None, None
        h_val = self.cbf.get_h_value(np.array([x_2d[0], x_2d[1], 0.0]))
        if self.cbf_active and t_clock > 0.5: # Delay CBF until robot enters 1.6 orbit
            A_t, b_t = self.cbf.get_constraints(np.pad(x_2d, (0,1)), np.pad(dx_2d, (0,1)), np.pad(u_ref, (0,1)), current_q_val)
            cbf_A, cbf_b = A_t[:, :2], b_t

        # 6. LEARNED INVERSION
        A_inv, b_learned = self.learned_model.get_inverse_dynamics(curr_q, curr_dq)
        
        # 7. QP CONSTRAINT SETUP
        bias_torque = A_inv @ (u_ref - b_learned)
        A_tau = np.vstack([A_inv, -A_inv])
        b_tau = np.hstack([self.tau_limits - bias_torque, self.tau_limits + bias_torque]).reshape(-1, 1)

        # 8. SOLVE QP
        if t_clock < 0.1:
            # Active Damping for very first few frames
            tau_cmd = -20.0 * curr_dq 
            mu = np.zeros(2)
        else:
            mu, feasible = solve_optimization(
                LfV, LgV, V, gamma, robust_clf_term=robust_term,
                torque_A=A_tau, torque_b=b_tau, cbf_A=cbf_A, cbf_b=cbf_b
            )
            if feasible:
                acc_cmd = u_ref + mu 
                tau_cmd = A_inv @ (acc_cmd - b_learned)
            else:
                tau_cmd = -15.0 * curr_dq 

        # 9. FRICTION NUDGE
        if 0.1 < t_clock < 1.0 and np.linalg.norm(curr_dq) < 1e-2:
            tau_cmd += 3.5 * np.sign(xd - x_2d) # Stronger nudge for 1.75 -> 1.6

        tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
        msg = Float64MultiArray(data=tau_cmd.tolist()); self.pub.publish(msg)

        # 10. LOGGING
        with self.lock:
            if len(self.log['t']) > 500: 
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock); self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
            self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
            self.log['h'].append(h_val); self.log['mu'].append(np.linalg.norm(mu))
            self.log['V'].append(V); self.log['err'].append(np.linalg.norm(x_2d - xd))

    def stop_robot(self):
        self.pub.publish(Float64MultiArray(data=[0.0]*2))

def main(args=None):
    rclpy.init(args=args)
    node = GazeboResclfNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # SETUP VISUALIZATION
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.8, 1])
    ax_traj = fig.add_subplot(gs[:, 0]); ax_h = fig.add_subplot(gs[0, 1])    
    ax_mu = fig.add_subplot(gs[1, 1]); ax_V = fig.add_subplot(gs[2, 1])    
    ax_err = fig.add_subplot(gs[3, 1])  
    plt.subplots_adjust(bottom=0.10, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = node.cbf.radii[0], node.cbf.radii[1], node.cbf.power_n
    x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    ax_traj.plot(x_b, y_b, 'g-', label='Safe Set')
    ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0); ax_traj.set_aspect('equal'); ax_traj.grid(True); ax_traj.legend()
    
    ln_h, = ax_h.plot([], [], 'g-'); ax_h.axhline(0, color='r', linestyle='--'); ax_h.set_title("Safety h(x)"); ax_h.grid(True)
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction ||μ||"); ax_mu.grid(True)
    ln_V, = ax_V.plot([], [], 'b-'); ax_V.set_title("Lyapunov V(x)"); ax_V.grid(True)
    ln_err, = ax_err.plot([], [], 'r-'); ax_err.set_title("Tracking Error ||e||"); ax_err.grid(True)

    check = CheckButtons(plt.axes([0.05, 0.02, 0.25, 0.06]), ['Safety On', 'Use Quantile'], [False, False])
    def toggle(label): 
        if label == 'Safety On': node.cbf_active = not node.cbf_active
        elif label == 'Use Quantile': node.use_quantile = not node.use_quantile
    check.on_clicked(toggle)

    def update(frame):
        with node.lock:
            if not node.log['t']: return ln_a, ln_t, ln_h, ln_mu, ln_V, ln_err
            t_d, x_d, y_d = node.log['t'], node.log['x'], node.log['y']
            xd_d, yd_d, h_d = node.log['xd'], node.log['yd'], node.log['h']
            mu_d, V_d, err_d = node.log['mu'], node.log['V'], node.log['err']

        ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
        ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
        ln_V.set_data(t_d, V_d); ln_err.set_data(t_d, err_d)
        
        for ax, data in zip([ax_h, ax_mu, ax_V, ax_err], [h_d, mu_d, V_d, err_d]):
            ax.set_xlim(t_d[0], t_d[-1])
            if data: ax.set_ylim(min(data)*0.9 - 0.1, max(data)*1.1 + 0.1)
        return ln_a, ln_t, ln_h, ln_mu, ln_V, ln_err

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
    node.stop_robot(); node.destroy_node(); rclpy.shutdown(); t_ros.join()

if __name__ == '__main__':
    main()




# """
# Main Node for 2-DOF Robot using GAZEBO Simulation
# CONTROLLER: Learned Model (Data-Driven Inverse Dynamics)
# PHYSICS/SENSING: Gazebo (True) + URDF (Kinematics only)
# """
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import pickle
# import os
# import sys
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import CheckButtons 

# # --- CONFIGURATIONS ---
# from ament_index_python.packages import get_package_share_directory

# # --- MODULAR IMPORTS ---
# from some_examples_py.CRCLF_CRCBF_2_link.robot_dynamics import RobotDynamics
# from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
# from some_examples_py.CRCLF_CRCBF_2_link.resclf_controller import RESCLF_Controller
# from some_examples_py.CRCLF_CRCBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
# from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

# URDF_CTRL = os.path.join(
#     get_package_share_directory("daadbot_desc"),
#     "urdf", "2_link_urdf", "2link_robot.urdf.xacro" 
# )

# # ROBUST PATH DETECTION (Finds /home/USER/xdaadbot_ws/...)
# MODEL_PATH = os.path.join(os.path.expanduser("~"), "xdaadbot_ws", "my_learned_robot.pkl")

# EE_NAMES = ["endEffector"]
# ALL_JOINTS = ["baseHinge", "interArm"]

# # --- 1. NEW HELPER CLASS FOR LEARNED MODEL ---
# class LearnedModelWrapper:
#     """ Loads the Scikit-Learn model and inverts it for control """
#     def __init__(self, model_path):
#         print(f"[LearnedModel] Loading from: {model_path}")
#         try:
#             with open(model_path, "rb") as f:
#                 data = pickle.load(f)
#                 self.model = data["model"]
#                 self.safety_bounds = data["safety_bounds"]
#             print(f"[LearnedModel] Success. Safety Bounds: {self.safety_bounds}")
#         except Exception as e:
#             print(f"[Error] Could not load model: {e}")
#             sys.exit(1)

#     def build_features(self, q, dq, tau):
#         # [FIX] UPDATED TO MATCH TRAINING SCRIPT (12 Features)
#         # Order: [tau1, tau2, dq1, dq2, s1, c1, s2, c2, s12, c12, dq1^2, dq2^2]
        
#         q1, q2 = q[0], q[1]
#         dq1, dq2 = dq[0], dq[1]
#         t1, t2 = tau[0], tau[1]
        
#         s1, c1 = np.sin(q1), np.cos(q1)
#         s2, c2 = np.sin(q2), np.cos(q2)
#         s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        
#         # New Squared Terms
#         dq1_sq = dq1**2
#         dq2_sq = dq2**2
        
#         # Must return (1, 12) array
#         return np.array([[t1, t2, dq1, dq2, s1, c1, s2, c2, s12, c12, dq1_sq, dq2_sq]])

#     def get_inverse_dynamics(self, q, dq):
#         """
#         Derives relationship: tau = A_inv * (acc - b)
#         by probing the learned black-box model.
#         """
#         # 1. Get Drift 'b' (Accel at Zero Torque)
#         feat_0 = self.build_features(q, dq, np.zeros(2)) # Shape (1, 12)
#         b = self.model.predict(feat_0)[0]
        
#         # 2. Get Control 'A' (Probe columns with Unit Torque)
#         feat_1 = self.build_features(q, dq, np.array([1.0, 0.0]))
#         col_1 = self.model.predict(feat_1)[0] - b
        
#         feat_2 = self.build_features(q, dq, np.array([0.0, 1.0]))
#         col_2 = self.model.predict(feat_2)[0] - b
        
#         A = np.column_stack([col_1, col_2])
        
#         # 3. Invert A to get mapping from Accel -> Torque
#         # Uses Pseudo-Inverse for numerical stability
#         A_inv = np.linalg.pinv(A)
#         return A_inv, b

# class GazeboResclfNode(Node):
#     def __init__(self):
#         super().__init__('gazebo_learned_ctrl_node')
        
#         # --- A. SETUP SENSING (URDF) ---
#         # We KEEP the URDF just to calculate Kinematics (Where am I?)
#         self.sensing_model = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        
#         # --- B. SETUP CONTROL (LEARNED MODEL) ---
#         # We USE this to calculate Torques (How do I move?)
#         self.learned_model = LearnedModelWrapper(MODEL_PATH)
        
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
#         self.cbf = CBF_SuperEllipsoid(
#             center=[0.0, 0.0, 0.0], 
#             lengths=[1.2, 1.2, 3.0], 
#             power_n=4, k_pos=21.0, k_vel=12.0
#         )
#         self.cbf_active = False 

#         # ROS Setup
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
#         self.timer = self.create_timer(1.0/100.0, self.control_loop) 
        
#         self.start_time = None
#         self.q = np.array([0.0, 0.1]) 
#         self.dq = np.zeros(2)
#         self.tau_limits = np.array([50.0, 30.0]) 

#         # Thread Lock for Safe Plotting
#         self.lock = threading.Lock()
#         self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

#     def cb_joints(self, msg):
#         try:
#             q_buf = [0.0] * 2; dq_buf = [0.0] * 2; found = 0
#             for i, name in enumerate(ALL_JOINTS):
#                 if name in msg.name:
#                     idx = msg.name.index(name)
#                     q_buf[i] = msg.position[idx]; dq_buf[i] = msg.velocity[idx]; found += 1
#             if found == 2: self.q = np.array(q_buf); self.dq = np.array(dq_buf)
#         except: pass

#     def control_loop(self):
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         # 1. SENSING (Use URDF ONLY for x, dx)
#         # Note: We ignore M, nle (Dynamics) from URDF here!
#         _, _, J, dJ, x, dx = self.sensing_model.compute_dynamics(self.q, self.dq)
#         J = J[0:2, :]; dJ = dJ[0:2, :]; x_2d = x[0:2]; dx_2d = dx[0:2]

#         # 2. TRAJECTORY (Elliptical)
#         xd_full, vd_full, ad_full = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
#         xd, vd, ad = xd_full[:2], vd_full[:2], ad_full[:2]

#         # 3. NOMINAL CONTROL
#         u_nominal = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
#         u_ref = ad + u_nominal 
        
#         # Robust CLF: Use learned safety bound
#         q_quantile = 0.0
#         LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
#             x_2d, dx_2d, xd, vd, q_quantile=q_quantile
#         )

#         # 4. CBF SAFETY
#         cbf_A, cbf_b = None, None
#         x_3d = np.array([x_2d[0], x_2d[1], 0.0])
#         h_val = self.cbf.get_h_value(x_3d)

#         if self.cbf_active:
#             dx_3d = np.array([dx_2d[0], dx_2d[1], 0.0])
#             u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
#             A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=q_quantile)
#             cbf_A = A_temp[:, :2]; cbf_b = b_temp

#         # --- KEY CHANGE: LEARNED DYNAMICS INVERSION ---
        
#         # 5. GET LEARNED MATRICES
#         # Relationship: tau = A_inv * (acc - b_learned)
#         A_inv, b_learned = self.learned_model.get_inverse_dynamics(self.q, self.dq)
        
#         # 6. QP CONSTRAINT FORMULATION
#         # We must constrain 'mu' (extra accel) so 'tau' fits limits.
#         # tau = A_inv * (u_ref + mu - b_learned)
        
#         # The torque needed for the reference motion alone
#         bias_torque = A_inv @ (u_ref - b_learned)
        
#         # The mapping from mu to torque
#         A_tau_base = A_inv 
        
#         # Constraint: A_tau * mu <= b_tau
#         A_tau = np.vstack([A_tau_base, -A_tau_base])
#         b_tau = np.hstack([
#             self.tau_limits - bias_torque,  # Upper Limit
#             self.tau_limits + bias_torque   # Lower Limit (flipped)
#         ]).reshape(-1, 1)

#         # 7. SOLVE QP
#         mu, feasible = solve_optimization(
#             LfV, LgV, V, gamma, 
#             robust_clf_term=robust_term,
#             torque_A=A_tau, torque_b=b_tau, 
#             cbf_A=cbf_A, cbf_b=cbf_b
#         )

#         if feasible:
#             # Apply Learned Control Law
#             acc_cmd = u_ref + mu 
#             tau_cmd = A_inv @ (acc_cmd - b_learned)
#         else:
#             tau_cmd = -5.0 * self.dq # Emergency Damping

#         tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
#         msg = Float64MultiArray(); msg.data = tau_cmd.tolist(); self.pub.publish(msg)

#         # LOGGING (Thread Safe)
#         with self.lock:
#             if len(self.log['t']) > 500: 
#                 for k in self.log: self.log[k].pop(0)
#             self.log['t'].append(t_clock)
#             self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
#             self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
#             self.log['h'].append(h_val); self.log['mu'].append(np.linalg.norm(mu))

#     def stop_robot(self):
#         msg = Float64MultiArray(data=[0.0]*2); self.pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = GazeboResclfNode()
    
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     fig = plt.figure(figsize=(10, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
#     ax_traj = fig.add_subplot(gs[:, 0]) 
#     ax_h = fig.add_subplot(gs[0, 1])    
#     ax_mu = fig.add_subplot(gs[1, 1])   
#     plt.subplots_adjust(bottom=0.15)
    
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual (Learned Ctrl)')
#     ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
#     # Safe Set
#     theta = np.linspace(0, 2*np.pi, 200)
#     rx, ry = node.cbf.radii[0], node.cbf.radii[1]
#     n  = node.cbf.power_n
#     x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
#     y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
#     ax_traj.plot(x_b, y_b, 'g-', label='Safe Set')
    
#     ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
#     ax_traj.set_aspect('equal', adjustable='box'); ax_traj.grid(True); ax_traj.legend()
#     ln_h, = ax_h.plot([], [], 'g-'); ax_h.axhline(0, color='r', linestyle='--'); ax_h.set_title("Safety h(x)"); ax_h.grid(True)
#     ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction ||μ||"); ax_mu.grid(True)

#     ax_check = plt.axes([0.05, 0.02, 0.15, 0.05]) 
#     check = CheckButtons(ax_check, ['Safety On'], [False])
#     def toggle(label): node.cbf_active = not node.cbf_active
#     check.on_clicked(toggle)

#     def update(frame):
#         # Thread Safe Reading
#         with node.lock:
#             if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
#             t_d = list(node.log['t'])
#             x_d = list(node.log['x']); y_d = list(node.log['y'])
#             xd_d = list(node.log['xd']); yd_d = list(node.log['yd'])
#             h_d = list(node.log['h']); mu_d = list(node.log['mu'])

#         ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
#         ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
#         if len(t_d) > 0:
#             ax_h.set_xlim(t_d[0], t_d[-1]); ax_mu.set_xlim(t_d[0], t_d[-1])
#             ax_h.set_ylim(-1.0, 1.0); ax_mu.set_ylim(-10.0, 20.0)
#         return ln_a, ln_t, ln_h, ln_mu

#     ani = FuncAnimation(fig, update, interval=50)
#     plt.show()
#     node.stop_robot(); node.destroy_node(); rclpy.shutdown(); t_ros.join()

# if __name__ == '__main__':
#     main()




# """
# Main Node for 2-DOF Robot using GAZEBO Simulation
# Updated with Dynamic Runtime Noise Injection for CR-CLF Testing
# """
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import CheckButtons 

# # --- CONFIGURATIONS ---
# from ament_index_python.packages import get_package_share_directory
# import os

# # --- MODULAR IMPORTS (Updated to CR versions) ---
# from some_examples_py.CRCLF_CRCBF_2_link.robot_dynamics import RobotDynamics
# from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
# from some_examples_py.CRCLF_CRCBF_2_link.resclf_controller import RESCLF_Controller
# from some_examples_py.CRCLF_CRCBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
# from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

# URDF_CTRL = os.path.join(
#     get_package_share_directory("daadbot_desc"),
#     "urdf",
#     "2_link_urdf", 
#     "2link_robot.urdf" 
# )

# EE_NAMES = ["endEffector"]
# ALL_JOINTS = ["baseHinge", "interArm"]

# # Set to the value output by your calibration script
# CALIBRATED_QUANTILE = 1000.0 

# class GazeboResclfNode(Node):
#     def __init__(self):
#         super().__init__('gazebo_crclf_node')
        
#         # --- 1. CONTROLLER SETUP ---
#         self.robot_ctrl = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
#         self.cbf = CBF_SuperEllipsoid(
#             center=[0.0, 0.0, 0.0], 
#             lengths=[1.2, 1.2, 3.0], 
#             power_n=4, k_pos=21.0, k_vel=12.0
#         )
#         self.cbf_active = False 
#         self.noise_active = False # [NEW] Noise Toggle
#         self.q_quantile = CALIBRATED_QUANTILE

#         # [NEW] Save baseline model masses so we don't drift to infinity
#         self.baseline_masses = [
#             self.robot_ctrl.model.inertias[i].mass 
#             for i in range(len(self.robot_ctrl.model.inertias))
#         ]

#         # --- 2. ROS INTERFACE ---
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
#         self.timer = self.create_timer(1.0/100.0, self.control_loop) 
        
#         # --- 3. STATE ---
#         self.start_time = None
#         self.q = np.array([0.0, 0.1]) 
#         self.dq = np.zeros(2)
#         self.tau_limits = np.array([50.0, 30.0]) 

#         self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

#     def cb_joints(self, msg):
#         try:
#             q_buf = [0.0] * 2
#             dq_buf = [0.0] * 2
#             found_count = 0
#             for i, name in enumerate(ALL_JOINTS):
#                 if name in msg.name:
#                     idx = msg.name.index(name)
#                     q_buf[i] = msg.position[idx]
#                     dq_buf[i] = msg.velocity[idx]
#                     found_count += 1
#             if found_count == 2:
#                 self.q = np.array(q_buf)
#                 self.dq = np.array(dq_buf)
#         except ValueError:
#             pass

#     def apply_dynamic_noise(self):
#         """
#         Injects real-time Gaussian noise into the controller's perception of the robot's mass.
#         Simulates unmodeled payloads or severe joint wear/tear.
#         """
#         if not self.noise_active:
#             # Restore to nominal state
#             for i in range(len(self.baseline_masses)):
#                 self.robot_ctrl.model.inertias[i].mass = self.baseline_masses[i]
#             return

#         # Up to +/- 30% dynamic variation at 100Hz!
#         for i in range(1, len(self.baseline_masses)):
#             noise_factor = np.random.uniform(100.01, 300.093)
#             self.robot_ctrl.model.inertias[i].mass = self.baseline_masses[i] * noise_factor

#     def control_loop(self):
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         # [NEW] Corrupt the model dynamically before computing dynamics
#         self.apply_dynamic_noise()

#         # A. DYNAMICS (Now using a noisy, inaccurate model)
#         M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(self.q, self.dq)
#         J = J[0:2, :]; dJ = dJ[0:2, :]; x_2d = x[0:2]; dx_2d = dx[0:2]

#         # B. TRAJECTORY
#         xd_full, vd_full, ad_full = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
#         xd, vd, ad = xd_full[:2], vd_full[:2], ad_full[:2]

#         # C. NOMINAL CONTROL
#         u_nominal = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
#         u_ref = ad + u_nominal 
        
#         # D. CR-CLF: Uses the quantile to buffer against the noise we just injected
#         LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
#             x_2d, dx_2d, xd, vd, q_quantile=self.q_quantile
#         )

#         # E. CR-CBF
#         cbf_A, cbf_b = None, None
#         x_3d = np.array([x_2d[0], x_2d[1], 0.0])
#         h_val = self.cbf.get_h_value(x_3d)

#         if self.cbf_active:
#             dx_3d = np.array([dx_2d[0], dx_2d[1], 0.0])
#             u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
#             A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=self.q_quantile)
#             cbf_A = A_temp[:, :2] 
#             cbf_b = b_temp

#         # F. QP SETUP
#         J_pinv = np.linalg.pinv(J)
#         drift_acc = u_ref - (dJ @ self.dq)
#         b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
#         A_tau_base = M @ J_pinv
#         A_tau = np.vstack([A_tau_base, -A_tau_base])
#         b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
#         b_tau = b_tau.reshape(-1, 1)

#         # G. SOLVE
#         mu, feasible = solve_optimization(
#             LfV, LgV, V, gamma, 
#             robust_clf_term=robust_term, 
#             torque_A=A_tau, torque_b=b_tau, 
#             cbf_A=cbf_A, cbf_b=cbf_b
#         )

#         if feasible:
#             acc_cmd = u_ref + mu 
#             tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle
#         else:
#             tau_cmd = -5.0 * self.dq + nle 

#         tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
#         msg = Float64MultiArray(); msg.data = tau_cmd.tolist(); self.pub.publish(msg)

#         # H. LOGGING
#         if len(self.log['t']) > 500: 
#             for k in self.log: self.log[k].pop(0)
            
#         self.log['t'].append(t_clock)
#         self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
#         self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
#         self.log['h'].append(h_val)
#         self.log['mu'].append(np.linalg.norm(mu))

#     def stop_robot(self):
#         msg = Float64MultiArray(data=[0.0]*2); self.pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = GazeboResclfNode()
    
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     fig = plt.figure(figsize=(10, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    
#     ax_traj = fig.add_subplot(gs[:, 0]) 
#     ax_h = fig.add_subplot(gs[0, 1])    
#     ax_mu = fig.add_subplot(gs[1, 1])   
#     plt.subplots_adjust(bottom=0.15)
    
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
#     ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
#     theta = np.linspace(0, 2*np.pi, 200)
#     rx, ry = node.cbf.radii[0], node.cbf.radii[1]
#     n  = node.cbf.power_n
#     x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
#     y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
#     ax_traj.plot(x_b, y_b, 'g-', label='Safe Set')
    
#     ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
#     ax_traj.set_aspect('equal', adjustable='box')
#     ax_traj.grid(True)
#     ax_traj.legend()

#     ln_h, = ax_h.plot([], [], 'g-'); ax_h.axhline(0, color='r', linestyle='--'); ax_h.set_title("Safety h(x)"); ax_h.grid(True)
#     ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction ||μ||"); ax_mu.grid(True)

#     # [NEW] Dual Checkbuttons for Safety and Noise
#     ax_check = plt.axes([0.05, 0.02, 0.25, 0.08]) 
#     check = CheckButtons(ax_check, ['Safety On', 'Dynamic Noise'], [False, False])
    
#     def toggle(label): 
#         if label == 'Safety On': node.cbf_active = not node.cbf_active
#         if label == 'Dynamic Noise': node.noise_active = not node.noise_active
#     check.on_clicked(toggle)

#     def update(frame):
#         if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
#         t_d = list(node.log['t'])
#         x_d = list(node.log['x']); y_d = list(node.log['y'])
#         xd_d = list(node.log['xd']); yd_d = list(node.log['yd'])
#         h_d = list(node.log['h']); mu_d = list(node.log['mu'])

#         ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
#         ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
        
#         if len(t_d) > 0:
#             ax_h.set_xlim(t_d[0], t_d[-1])
#             ax_mu.set_xlim(t_d[0], t_d[-1])
#             ax_h.set_ylim(-1.0, 1.0)
#             ax_mu.set_ylim(-10.0, 20.0)

#         return ln_a, ln_t, ln_h, ln_mu

#     ani = FuncAnimation(fig, update, interval=50)
#     plt.show()
    
#     node.stop_robot(); node.destroy_node(); rclpy.shutdown(); t_ros.join()

# if __name__ == '__main__':
#     main()