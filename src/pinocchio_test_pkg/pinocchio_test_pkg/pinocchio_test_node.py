#!/usr/bin/env python3.10

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import pinocchio as pin
import numpy as np  # <-- important
from pinocchio.utils import rand

class PinocchioTestNode(Node):
    def __init__(self):
        super().__init__('pinocchio_test')

        # Subscribe to joint_states
        self.subscription = self.create_subscription(
            JointState, 'joint_states', self.listener_callback, 10
        )

        # Create a minimal 2-joint model
        self.model = pin.Model()
        joint1_id = self.model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), 'joint1')
        joint2_id = self.model.addJoint(joint1_id, pin.JointModelRY(), pin.SE3.Identity(), 'joint2')
        # Add an end-effector frame at joint2
        self.model.addFrame(pin.Frame('ee', joint2_id, pin.SE3.Identity(), pin.FrameType.BODY))
        # Create data container for computations
        self.data = self.model.createData()

    def listener_callback(self, msg: JointState):
        if len(msg.position) >= self.model.nq:
            # Convert joint positions to NumPy array (required by Pinocchio)
            q = np.array(msg.position[:self.model.nq], dtype=np.float64)
            # Compute forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            ee_frame = self.data.oMf[self.model.getFrameId('ee')]
            # Log end-effector transform
            self.get_logger().info(f'End-effector frame:\n{ee_frame}')

def main(args=None):
    rclpy.init(args=args)
    node = PinocchioTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
