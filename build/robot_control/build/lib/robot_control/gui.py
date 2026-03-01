import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class GuiNode(Node):
    def __init__(self):
        super().__init__('gui_node')

        # Publisher to send commands to the arm
        self.start_publisher = self.create_publisher(Bool, '/start', 10)

        # Simulate correction with a timer (for demo)
        self.timer = self.create_timer(1.0, self.correction_callback)  # Run every 1 second

    def correction_callback(self):
        # Replace this with actual correction logic
        start_msg = Bool()
        start_msg.data = True  # Simulated start command

        # Publish the start command
        self.start_publisher.publish(start_msg)
        self.get_logger().info(f"Published Start Command: {start_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = GuiNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()