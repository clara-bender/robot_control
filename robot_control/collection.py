import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray, Int32
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME

class CollectionNode(Node):
    def __init__(self):
        super().__init__('collection_node')
        # Inputs
        self.REPO_NAME = "clara/inference_collection"
        self.FPS_COLLECT = 20
        self.DT_COLLECT = 1.0 / self.FPS_COLLECT # multiple of 10

        self.TASK_DESCRIPTION = "Pick up the bag and place it on the blue x"

        # Subscriber: master_controller.py, "start_collection" signal
        self.start_collection_sub = self.create_subscription(Bool, '/start_collection', self.start_collection_callback, 10)
        
        # Subscribers: gui.py, save or discard data
        self.save_sub = self.create_subscription(Bool, '/save_button', self.save_callback, 10)
        self.discard_sub = self.create_subscription(Bool, '/discard_button', self.discard_callback, 10)

        # Subscribers: gui.py, failure or success signal
        self.failure_success_sub = self.create_subscription(Bool, '/failure_success_button', self.failure_success_callback, 10)
        
        # Scubscribers: master_controller.py, current robot state
        self.servo_state_sub = self.create_subscription(Float32MultiArray, '/servo_state', self.servo_state_callback, 10)
        self.gripper_state_sub = self.create_subscription(Int32, '/gripper_state', self.gripper_state_callback, 10)
        
        # Subscribers: cameras.py, current camera images
        self.wrist_sub = self.create_subscription(Image, '/camera_image/wrist', self.wrist_camera_callback, 10)
        self.tripod_sub = self.create_subscription(Image, '/camera_image/tripod', self.tripod_camera_callback, 10)

        # Initializations
        self.start_collection = False
        self.servo_state = None
        self.gripper_state = None
        self.wrist_camera = None
        self.tripod_camera = None
        self.prev_data = None
        self.save_data = False
        self.discard_data = False
        self.frames_recorded = 0
        self.failure = False

        self.dataset_path = HF_LEROBOT_HOME / self.REPO_NAME

        if self.dataset_path.exists(): 
            self.dataset = LeRobotDataset(
                root=self.dataset_path,
                repo_id=self.REPO_NAME,
            )
            print("Adding to existing dataset, waiting for signal.")
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=self.REPO_NAME,
                robot_type="xarm",
                fps=self.FPS_COLLECT,
                features={
                    "exterior_image_1_left": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "exterior_image_2_left": { # this one is not used, put it as zeros or something
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image_left": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "joint_position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["joint_position"],
                    },
                    "gripper_position": {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": ["gripper_position"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                        "names": ["actions"],
                    },
                },
            )

        # Collect data
        self.timer = self.create_timer(self.DT_COLLECT, self.collection_callback)  # Run every 1 second

    def failure_success_callback(self, msg):
        self.failure = msg.data
    
    def start_collection_callback(self, msg):
        self.start_collection = msg.data

    def servo_state_callback(self, msg):
        self.servo_state = msg.data

    def gripper_state_callback(self, msg):
        self.gripper_state = msg.data

    def wrist_camera_callback(self, msg):
        bridge = CvBridge()
        self.wrist_camera = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def tripod_camera_callback(self, msg):
        bridge = CvBridge()
        self.tripod_camera = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def save_callback(self, msg):
        save_data = msg.data
        if save_data and self.prev_data is not None:
            self.dataset.save_episode()
            self.get_logger().info(f"Episode saved with {self.frames_recorded} frames.")
            self.frames_recorded = 0
            self.prev_data = None

    def discard_callback(self, msg):
        discard_data = msg.data
        if discard_data and self.prev_data is not None:
            # HARD RESET: clears the in-memory episode buffer
            self.dataset = LeRobotDataset(root=self.dataset_path,repo_id=self.REPO_NAME,)
            self.get_logger().info("Episode discarded, buffer cleared.")
            self.prev_data = None

    def collection_callback(self):

        if not self.start_collection:
            return
        
        self.get_logger().info("Collecting data...")
        
        # 1. Capture CURRENT state (Time t+1 relative to prev_data)
        #joints = arm.get_servo_angle(is_radian=True)[1][:6]
        pose = self.servo_state.tolist()
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        # ensure roll and yaw are continuous, also make sure pitch doesn't exceed 90 deg
        # when collecting demos
        angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    
        gripper = (self.gripper_state - 850) / -860
        curr_state = np.array(pose[:3] + angles_rad + [gripper], dtype=np.float32)
        
        if self.failure:
            base2 = np.ones_like(self.tripod_camera) * 255
        else:
            base2 = np.zeros_like(self.tripod_camera)

        # 2. If we have a previous observation, record it with CURRENT state as the action
        if self.prev_data is not None:
            self.dataset.add_frame(
                {
                    "joint_position": self.prev_data["joints"],
                    "gripper_position": self.prev_data["gripper"],
                    "actions": curr_state,  # This is the "future" state reached
                    "exterior_image_1_left": self.prev_data["base"],
                    "exterior_image_2_left": self.prev_data["base2"],
                    "wrist_image_left": self.prev_data["wrist"],
                    "task": self.TASK_DESCRIPTION,
                }
            )
            self.frames_recorded += 1

        # 3. Store current observations to be paired with the next frame's state
        self.prev_data = {
            "joints": curr_state[:6],
            "gripper": curr_state[-1:],
            "wrist": self.wrist_camera,
            "base": self.tripod_camera,
            "base2": base2,
        }

        self.get_logger().info(f"Failure: {self.failure}")

def main(args=None):
    rclpy.init(args=args)
    node = CollectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()