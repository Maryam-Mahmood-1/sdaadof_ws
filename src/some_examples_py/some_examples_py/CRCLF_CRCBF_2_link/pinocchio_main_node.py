"""
Main Node for 2-DOF Robot: Parallel Comparison (Regular vs Robust)
FEATURE: Simultaneous simulation and dual-plotting
"""
import numpy as np
import pinocchio as pin
import threading
import time
import pickle
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
from some_examples_py.CRCLF_CRCBF_2_link.resclf_controller import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

URDF_PATH = os.path.expanduser("~/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf")
MODEL_PATH = os.path.join(os.path.expanduser("~"), "xdaadbot_ws", "my_learned_robot2.pkl")

class PinocchioRobotWrapper:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId("endEffector")

    def get_state_and_kinematics(self, q, dq):
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        x = self.data.oMf[self.ee_id].translation
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return x[0:2], (J @ dq)[0:2]

class LearnedModelWrapper:
    def __init__(self, model_path):
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.safety_bounds = data["safety_bounds"]
        except Exception as e:
            print(f"[Error] {e}"); sys.exit(1)

    def build_features(self, q, dq, tau):
        s1, c1, s2, c2 = np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([[tau[0], tau[1], dq[0], dq[1], s1, c1, s2, c2, s12, c12, dq[0]**2, dq[1]**2]])

    def get_inverse_dynamics(self, q, dq):
        feat_0 = self.build_features(q, dq, np.zeros(2))
        b = self.model.predict(feat_0)[0]
        feat_1 = self.build_features(q, dq, np.array([1.0, 0.0]))
        col_1 = self.model.predict(feat_1)[0] - b
        feat_2 = self.build_features(q, dq, np.array([0.0, 1.0]))
        col_2 = self.model.predict(feat_2)[0] - b
        return np.linalg.pinv(np.column_stack([col_1, col_2])), b

class PinocchioSimNode:
    def __init__(self):
        self.robot = PinocchioRobotWrapper(URDF_PATH)
        self.learned_model = LearnedModelWrapper(MODEL_PATH)
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.2, 1.2, 3.0])
        
        # Parallel States: [0] is Regular, [1] is Robust (Quantile)
        self.q = [np.array([0.0, 0.1]), np.array([0.0, 0.1])]
        self.dq = [np.zeros(2), np.zeros(2)]
        
        self.dt = 0.008
        self.t_clock = 0.0
        self.tau_limits = np.array([50.0, 30.0])
        self.cbf_active = False 
        self.learned_quantile_val = self.learned_model.safety_bounds['x']
        
        self.lock = threading.Lock()
        # Log keys now store tuples or separate lists for (Regular, Robust)
        self.log = {'t':[], 'xd':[], 'yd':[], 
                    'x0':[], 'y0':[], 'h0':[], 'mu0':[], 'V0':[], 'err0':[],
                    'x1':[], 'y1':[], 'h1':[], 'mu1':[], 'V1':[], 'err1':[]}

    def physics_step(self):
        # 1. Get Trajectory (Shared reference)
        # Using state from Regular robot to drive trajectory generator logic
        x_ref_sense, _ = self.robot.get_state_and_kinematics(self.q[0], self.dq[0])
        xd_full, vd_full, ad_full = self.traj_gen.get_ref(self.t_clock, current_actual_pos=np.pad(x_ref_sense, (0,1)))
        xd, vd, ad = xd_full[:2], vd_full[:2], ad_full[:2]

        results = []

        # 2. Compute for both Regular (q=0) and Robust (q=quantile)
        for i in range(2):
            q_val = self.learned_quantile_val if i == 1 else 0.0
            qi, dqi = self.q[i], self.dq[i]
            
            x_2d, dx_2d = self.robot.get_state_and_kinematics(qi, dqi)
            u_ref = ad + self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
            LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(x_2d, dx_2d, xd, vd, q_quantile=q_val*15.0)

            cbf_A, cbf_b = None, None
            h_val = self.cbf.get_h_value(np.pad(x_2d, (0,1)))
            if self.cbf_active:
                A_t, b_t = self.cbf.get_constraints(np.pad(x_2d, (0,1)), np.pad(dx_2d, (0,1)), np.pad(u_ref, (0,1)), q_quantile=q_val)
                cbf_A, cbf_b = A_t[:, :2], b_t

            A_inv, b_learned = self.learned_model.get_inverse_dynamics(qi, dqi)
            bias = A_inv @ (u_ref - b_learned)
            A_tau = np.vstack([A_inv, -A_inv])
            b_tau = np.hstack([self.tau_limits - bias, self.tau_limits + bias]).reshape(-1, 1)

            mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_term, A_tau, b_tau, cbf_A, cbf_b)

            tau_cmd = A_inv @ (u_ref + mu - b_learned) if feasible else -5.0 * dqi
            tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)

            # Integrate
            ddq = pin.aba(self.robot.model, self.robot.data, qi, dqi, tau_cmd)
            self.dq[i] += ddq * self.dt
            self.q[i] = pin.integrate(self.robot.model, qi, self.dq[i] * self.dt)
            
            results.append((x_2d, h_val, np.linalg.norm(mu), V, np.linalg.norm(x_2d - xd)))

        # 3. Logging
        with self.lock:
            if len(self.log['t']) > 500: 
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(self.t_clock)
            self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
            for i, res in enumerate(results):
                self.log[f'x{i}'].append(res[0][0]); self.log[f'y{i}'].append(res[0][1])
                self.log[f'h{i}'].append(res[1]); self.log[f'mu{i}'].append(res[2])
                self.log[f'V{i}'].append(res[3]); self.log[f'err{i}'].append(res[4])

        self.t_clock += self.dt

def main():
    sim = PinocchioSimNode()
    threading.Thread(target=lambda: [sim.physics_step() or time.sleep(sim.dt) for _ in iter(int, 1)], daemon=True).start()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.6, 1])
    ax_traj = fig.add_subplot(gs[:, 0])
    axes_side = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    
    # Trajectory Lines
    ln_reg_traj, = ax_traj.plot([], [], 'r-', label='Regular (q=0)', alpha=0.6)
    ln_rob_traj, = ax_traj.plot([], [], 'b-', label='Robust (q=learned)', alpha=0.8)
    ln_ref_traj, = ax_traj.plot([], [], 'k--', label='Reference', linewidth=1)
    
    # Boundary Plotting
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = sim.cbf.radii[0], sim.cbf.radii[1], sim.cbf.power_n
    xb = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta))**(2/n))
    yb = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta))**(2/n))
    ax_traj.plot(xb, yb, 'g-', label='Safe Set', linewidth=2)
    
    ax_traj.set_xlim(-1.8, 1.8); ax_traj.set_ylim(-1.8, 1.8)
    ax_traj.set_aspect('equal'); ax_traj.grid(True); ax_traj.legend(loc='upper right')

    # Side Plots Lines
    side_keys = ['h', 'mu', 'V', 'err']
    side_titles = ["Safety h(x)", "Correction ||μ||", "Lyapunov V(x)", "Tracking Error"]
    side_lines = []
    for i, ax in enumerate(axes_side):
        l_reg, = ax.plot([], [], 'r-', alpha=0.5)
        l_rob, = ax.plot([], [], 'b-', alpha=0.8)
        side_lines.append((l_reg, l_rob))
        ax.set_title(side_titles[i]); ax.grid(True)
    axes_side[0].axhline(0, color='grey', linestyle='--')

    ax_check = plt.axes([0.7, 0.02, 0.2, 0.05])
    check = CheckButtons(ax_check, ['Safety Active'], [False])
    def toggle(label): sim.cbf_active = not sim.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        with sim.lock:
            if not sim.log['t']: return
            t = sim.log['t']
            ln_reg_traj.set_data(sim.log['x0'], sim.log['y0'])
            ln_rob_traj.set_data(sim.log['x1'], sim.log['y1'])
            ln_ref_traj.set_data(sim.log['xd'], sim.log['yd'])
            
            for i, (l_reg, l_rob) in enumerate(side_lines):
                key = side_keys[i]
                dr, db = sim.log[f'{key}0'], sim.log[f'{key}1']
                l_reg.set_data(t, dr); l_rob.set_data(t, db)
                axes_side[i].set_xlim(t[0], t[-1])
                axes_side[i].set_ylim(min(min(dr), min(db))-0.1, max(max(dr), max(db))*1.1+0.1)

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()