import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class CorrectionNode(Node):
    def __init__(self):
        super().__init__('correction_node')

        # Publisher to send commands to the arm
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_manual', 10)
        self.mode_publisher = self.create_publisher(Bool, '/cmd_mode', 10)

        # Simulate correction with a timer (for demo)
        self.timer = self.create_timer(1.0, self.correction_callback)  # Run every 1 second

    def correction_callback(self):
        # Replace this with actual correction logic
        cmd_msg = Twist()
        mode_msg = Bool()

        # Example: Inference might output a velocity command
        cmd_msg.linear.x = 0.5  # Simulated output from inference model
        
        cmd_msg.linear.y = 2.0
        cmd_msg.linear.z = 2.0

        mode_msg.data = False  # Simulated mode switch to manual for correction

        # Publish the command
        self.cmd_publisher.publish(cmd_msg)
        self.get_logger().info(f"Published Correction Command: {cmd_msg.linear.x}, {cmd_msg.linear.y}, {cmd_msg.linear.z}")

        self.mode_publisher.publish(mode_msg)
        self.get_logger().info(f"Published Correction Mode: {mode_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = CorrectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()