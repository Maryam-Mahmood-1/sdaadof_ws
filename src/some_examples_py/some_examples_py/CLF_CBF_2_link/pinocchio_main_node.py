"""
Main Node for 2-DOF Robot using Adaptive QP Solver
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 

# --- CONFIGURATIONS ---
from ament_index_python.packages import get_package_share_directory
import os

# --- MODULAR IMPORTS ---
# Ensure these match your folder structure
from some_examples_py.CLF_CBF_2_link.robot_dynamics import RobotDynamics
from some_examples_py.CLF_CBF_2_link.trajectory_generator import TrajectoryGenerator
from some_examples_py.CLF_CBF_2_link.resclf_controller import RESCLF_Controller
from some_examples_py.CLF_CBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CLF_CBF_2_link.qp_solver import solve_optimization  # IMPORTING NEW SOLVER

URDF_PHYSICS = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "2_link_urdf", 
    "2link_robot.urdf.xacro" 
)

URDF_CTRL = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "2_link_urdf",
    "2link_robot.urdf.xacro" 
)

EE_NAMES = ["endEffector"]
ALL_JOINTS = ["baseHinge", "interArm"]

class RealTimePhysicsNode(Node):
    def __init__(self):
        super().__init__('rt_physics_sim_node')
        
        # --- 1. CONTROLLER SETUP ---
        self.robot_ctrl = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.traj_gen = TrajectoryGenerator() # Ensure this has your ellipse parameters (1.6, 0.9)
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
        # Safety Barrier (Planar Ellipse)
        # We define it in 3D but will check slices.
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.0], 
            lengths=[1.2, 1.2, 3.0], 
            power_n=4, k_pos=10.0, k_vel=10.0
        )
        self.cbf_active = False 

        # --- 2. PHYSICS ENGINE SETUP ---
        self.model_phys = pin.buildModelFromUrdf(URDF_PHYSICS)
        self.data_phys = self.model_phys.createData()
        self.phys_joint_ids = [self.model_phys.getJointId(name) for name in ALL_JOINTS]

        # --- 3. STATE INITIALIZATION ---
        # Initial config to start inside the safe zone (1.2m radius)
        q_init = np.array([0.0, 0.0]) 
        # q_init = np.array([0.0, -3.14159])

        self.q_sim = pin.neutral(self.model_phys) 
        self.v_sim = np.zeros(self.model_phys.nv)
        
        for i, jid in enumerate(self.phys_joint_ids):
             idx_q = self.model_phys.joints[jid].idx_q
             self.q_sim[idx_q] = q_init[i]

        self.lock = threading.Lock()
        self.q_read = q_init.copy()
        self.dq_read = np.zeros(2)
        self.tau_command = np.zeros(2)
        self.tau_limits = np.array([15.0, 15.0]) 

        # --- 4. THREADING & ROS ---
        self.running = True
        self.dt_phys = 0.001 
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()

        self.control_rate = 100.0
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None
        
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

    def physics_loop(self):
        """ The Physics Engine Thread (Simulating the Hardware) """
        print("--- Physics Engine Started ---")
        next_tick = time.time()
        
        while self.running:
            with self.lock:
                current_tau = self.tau_command.copy()

            tau_full = np.zeros(self.model_phys.nv)
            damping = 0.1 * self.v_sim 
            
            for i, jid in enumerate(self.phys_joint_ids):
                idx_v = self.model_phys.joints[jid].idx_v
                tau_full[idx_v] = current_tau[i] - damping[idx_v]

            try:
                ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_full)
                self.v_sim += ddq * self.dt_phys
                self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
            except Exception as e:
                print(f"Physics Error: {e}")

            q_sys = np.zeros(2)
            dq_sys = np.zeros(2)
            for i, jid in enumerate(self.phys_joint_ids):
                q_sys[i] = self.q_sim[self.model_phys.joints[jid].idx_q]
                dq_sys[i] = self.v_sim[self.model_phys.joints[jid].idx_v]

            with self.lock:
                self.q_read = q_sys
                self.dq_read = dq_sys

            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def control_loop(self):
        """ The Control Brain """
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        with self.lock:
            q_c = self.q_read.copy()
            dq_c = self.dq_read.copy()

        # 1. DYNAMICS (Full from Robot)
        M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(q_c, dq_c)
        
        # --- DATA SLICING (Crucial Step) ---
        # The robot is planar. We MUST slice 3D/6D vectors to 2D (x, y) 
        # so the solver receives consistent dimensions.
        
        # J is (6, 2). Take top 2 rows (linear x, y)
        J = J[0:2, :]   
        dJ = dJ[0:2, :]
        
        # State is (3,). Take first 2.
        x_2d = x[0:2]
        dx_2d = dx[0:2]

        # 2. TRAJECTORY GENERATION
        # Pass 3D padded state to generator, receive 3D ref
        xd_full, vd_full, ad_full = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
        
        # Slice Reference to 2D
        xd = xd_full[:2]
        vd = vd_full[:2]
        ad = ad_full[:2]

        # 3. NOMINAL CONTROL (CLF)
        # Everything here is 2D
        u_nominal = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
        u_ref = ad + u_nominal 
        LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x_2d, dx_2d, xd, vd)
        
        # 4. SAFETY BARRIER (CBF)
        cbf_A, cbf_b = None, None
        
        # Evaluation requires 3D point (append z=0)
        x_3d = np.array([x_2d[0], x_2d[1], 0.0])
        h_val = self.cbf.get_h_value(x_3d)

        # Safety Check: Don't activate if we spawn inside the wall
        if self.cbf_active and h_val < 0.0:
            print(f"[WARN] Robot unsafe (h={h_val:.2f}). Disabling CBF.")
            self.cbf_active = False

        if self.cbf_active:
            dx_3d = np.array([dx_2d[0], dx_2d[1], 0.0])
            u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
            
            # Get 3D constraints
            A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d)
            
            # Slice to 2D columns (ignore Z influence)
            cbf_A = A_temp[:, :2] 
            cbf_b = b_temp

        # 5. TORQUE MAPPING
        J_pinv = np.linalg.pinv(J)
        drift_acc = u_ref - (dJ @ dq_c)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        
        # Torque Limits Constraint: A_tau * mu <= b_tau
        # This maps task acc (mu) to joint torques
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

        # 6. SOLVE QP (EXTERNAL FILE)
        # LgV is (1,2). The solver will detect this and solve for mu in R^2.
        mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau, cbf_A=cbf_A, cbf_b=cbf_b)

        # 7. APPLY RESULT
        if feasible:
            # Reconstruct Torque: tau = M * J_inv * (u_ref + mu - drift) + nle
            acc_cmd = u_ref + mu 
            tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_c))) + nle
        else:
            # Fallback: Simple braking
            tau_out = -5.0 * dq_c + nle 

        tau_out = np.clip(tau_out, -self.tau_limits, self.tau_limits)
        mu_norm = np.linalg.norm(mu)

        with self.lock:
            self.tau_command = tau_out
            
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock)
            self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
            self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
            self.log['h'].append(h_val)
            self.log['mu'].append(mu_norm)

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePhysicsNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # PLOTTING
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    
    ax_traj = fig.add_subplot(gs[:, 0]) 
    ax_h = fig.add_subplot(gs[0, 1])
    ax_mu = fig.add_subplot(gs[1, 1])
    
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual') 
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry = node.cbf.radii[0], node.cbf.radii[1]
    cx, cy = node.cbf.center[0], node.cbf.center[1]
    n = node.cbf.power_n
    x_bound = cx + rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_bound = cy + ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    
    ax_traj.plot(x_bound, y_bound, 'g-', label='Safe Set')
    ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
    ax_traj.set_aspect('equal')
    ax_traj.legend(loc='upper right')
    ax_traj.grid(True)

    ln_h, = ax_h.plot([], [], 'g-', linewidth=1.5)
    ax_h.set_title("Safety h(x)")
    ax_h.set_ylim(-1.0, 2.0)
    ax_h.axhline(0, color='r', linestyle='--')
    ax_h.grid(True)

    ln_mu, = ax_mu.plot([], [], 'k-', linewidth=1.5)
    ax_mu.set_title("Correction ||μ||")
    ax_mu.set_ylim(0, 20.0) 
    ax_mu.grid(True)

    ax_check = plt.axes([0.05, 0.02, 0.15, 0.05]) 
    check = CheckButtons(ax_check, ['Safety On'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        with node.lock:
            if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
            t_d = list(node.log['t'])
            x_d = list(node.log['x']); y_d = list(node.log['y'])
            xd_d = list(node.log['xd']); yd_d = list(node.log['yd'])
            h_d = list(node.log['h']); mu_d = list(node.log['mu'])

        ln_a.set_data(x_d, y_d) 
        ln_t.set_data(xd_d, yd_d)
        ln_h.set_data(t_d, h_d)
        ln_mu.set_data(t_d, mu_d)
        
        if len(t_d) > 0:
            window_start = max(0, t_d[-1] - 10)
            window_end = t_d[-1] + 1
            ax_h.set_xlim(window_start, window_end)
            ax_mu.set_xlim(window_start, window_end)
        
        return ln_a, ln_t, ln_h, ln_mu

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
    
    node.running = False
    node.phys_thread.join()
    node.destroy_node()
    rclpy.shutdown()
    t_ros.join()

if __name__ == '__main__':
    main()