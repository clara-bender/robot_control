import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # Publisher to send commands to the arm
        self.publisher = self.create_publisher(Twist, '/cmd_auto', 10)

        # Simulate inference with a timer (for demo)
        self.timer = self.create_timer(1.0, self.inference_callback)  # Run every 1 second

    def inference_callback(self):
        # Replace this with actual inference logic
        msg = Twist()

        # Example: Inference might output a velocity command
        msg.linear.x = 0.5  # Simulated output from inference model
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        # Publish the command
        self.publisher.publish(msg)
        self.get_logger().info(f"Published Inference Command: {msg.linear.x}, {msg.linear.y}, {msg.linear.z}")

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()