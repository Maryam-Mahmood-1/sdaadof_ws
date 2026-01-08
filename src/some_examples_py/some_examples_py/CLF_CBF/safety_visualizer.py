import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class SafetyVisualizer(Node):
    def __init__(self):
        super().__init__('safety_visualizer')
        self.pub = self.create_publisher(Marker, '/safety_marker', 10)
        self.timer = self.create_timer(0.5, self.publish_marker)
        self.get_logger().info("Transparent Safety Visualizer Started")

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        # --- CHANGES FOR SEE-THROUGH RECTANGLE ---
        marker.ns = "safe_cage_plane"
        marker.id = 0
        marker.type = Marker.CUBE  # Use CUBE to make a filled shape
        marker.action = Marker.ADD
        
        # 1. Define Position (Center of the rectangle)
        # Range X: -0.3 to 0.3  -> Center = 0.0
        # Range Y: -0.24 to 0.24 -> Center = 0.0
        # Z = 0.72
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.72
        
        # Orientation (Flat)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # 2. Define Scale (Dimensions of the rectangle)
        # Width X = 0.3 - (-0.3) = 0.6
        # Height Y = 0.24 - (-0.24) = 0.48
        # Thickness Z = Small value to look like a plane
        marker.scale.x = 0.6
        marker.scale.y = 0.48
        marker.scale.z = 0.01  # 1cm thick glass look
        
        # 3. Define Color (Green + Alpha)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3   # <--- ALPHA: 0.3 = 30% Visible (See-through)
        
        self.pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()