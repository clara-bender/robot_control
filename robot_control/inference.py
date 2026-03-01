import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from cv_bridge import CvBridge

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # =========================
        # Policy Setup
        # =========================
        config = _config.get_config("pi05_xarm_finetune")
        checkpoint_dir = download.maybe_download(
            "/home/admin/openpi/checkpoints/pi05_xarm_finetune/clara_training1/25000"
        )
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # Camera Subscriptions
        self.wrist_sub = self.create_subscription(Image, '/camera_image/wrist', self.wrist_camera_callback, 10)
        self.tripod_sub = self.create_subscription(Image, '/camera_image/tripod', self.tripod_camera_callback, 10)

        # Publisher to send commands to the arm
        self.publisher = self.create_publisher(Twist, '/cmd_auto', 10)

        # Simulate inference with a timer (for demo)
        self.timer = self.create_timer(1.0, self.inference_callback)  # Run every 1 second

    def wrist_camera_callback(self, msg):
        bridge = CvBridge()
        self.wrist_camera = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def tripod_camera_callback(self, msg):
        bridge = CvBridge()
        self.tripod_camera = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

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