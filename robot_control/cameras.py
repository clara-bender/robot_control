import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import pyrealsense2 as rs
#from robot_control.msg import MultiImage
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Confirmed working :)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        # =========================
        # RealSense Camera Setup
        # =========================
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) < 2:
            raise RuntimeError("Need at least two RealSense cameras connected")

        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        print("Found cameras:", serials)

        self.pipelines = []
        self.configs = []

        for serial in serials:
            pipeline = rs.pipeline()
            config_rs = rs.config()
            config_rs.enable_device(serial)
            config_rs.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
            pipeline.start(config_rs)
            self.pipelines.append(pipeline)
            self.configs.append(config_rs)

        # Publisher
        self.wrist_pub = self.create_publisher(Image, '/camera_image/wrist', 10)
        self.tripod_pub = self.create_publisher(Image, '/camera_image/tripod', 10)

        self.publish_images()

    def publish_images(self):
        while True:
            # Get frames from both cameras, includes color, depth, infrared etc.
            wrist_img = self.pipelines[1].wait_for_frames()
            tripod_img = self.pipelines[0].wait_for_frames()

            # Extract color frames
            wrist_color = wrist_img.get_color_frame()
            tripod_color = tripod_img.get_color_frame()

            # Convert to numpy arrays
            wrist_color_np = np.asanyarray(wrist_color.get_data())
            tripod_color_np = np.asanyarray(tripod_color.get_data())

            # Convert to ROS Image messages
            bridge = CvBridge()
            wrist_msg = bridge.cv2_to_imgmsg(wrist_color_np, encoding="rgb8")
            tripod_msg = bridge.cv2_to_imgmsg(tripod_color_np, encoding="rgb8")

            # Publish
            self.tripod_pub.publish(tripod_msg)
            print("Tripod camera shape: ",tripod_color_np.shape)
            self.wrist_pub.publish(wrist_msg)
            print("Wrist camera shape: ", wrist_color_np.shape)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()