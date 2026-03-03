import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, Float32MultiArray
from xarm.wrapper import XArmAPI
import time
import numpy as np

class MasterController(Node):
    def __init__(self):
        super().__init__('master_controller')

        MASTER_HZ = 30.0
        self.get_logger().info(f'Master Controller initialized, running at {MASTER_HZ} Hz')

        # Start up XArm
        self.arm = XArmAPI("192.168.1.222")
        self.arm.connect()
        if self.arm.get_state() != 0:
            self.arm.clean_error()
            time.sleep(0.5)
        self.arm.motion_enable(enable=True)
        self.arm.set_gripper_enable(enable=True) 
        self.arm.set_gripper_mode(0)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(2.0)

        gripper_position = self.arm.get_gripper_position()[1]
        print(f"Gripper position: {gripper_position}")
        
        # Subscriber: gui.py start button
        self.start_sub = self.create_subscription(Bool,'/start_button',self.start_callback,10)

        # Subscriber: correction.py, mode toggle (left and right button), True for manual, False for auto
        self.mode_sub = self.create_subscription(Bool,'/manual_mode',self.mode_callback,10)

        # Subscriber: inference.py, "auto" commands
        self.auto_gripper_sub = self.create_subscription(Int32,'/cmd_auto_gripper',self.auto_gripper_callback,10)
        self.auto_servo_sub = self.create_subscription(Float32MultiArray,'/cmd_auto_servo',self.auto_servo_callback,10)

        # Subscriber: correction.py, "manual" commands
        self.manual_gripper_sub = self.create_subscription(Int32,'/cmd_manual_gripper',self.manual_gripper_callback,10)
        self.manual_servo_sub = self.create_subscription(Float32MultiArray,'/cmd_manual_servo',self.manual_servo_callback,10)

        # Publisher: Current robot state
        self.servo_state_pub = self.create_publisher(Float32MultiArray, '/servo_state', 10)
        self.gripper_state_pub = self.create_publisher(Int32, '/gripper_state', 10)

        # Publisher: indicate if collection should start (for correction.py)
        self.start_collection_pub = self.create_publisher(Bool, '/start_collection', 10)

        # Initializations
        self.start = False
        self.current_auto_gripper_cmd = None
        self.current_auto_servo_cmd = None
        self.current_manual_gripper_cmd = None
        self.current_manual_servo_cmd = None
        self.mode = 'auto'  # start in auto mode

        # Timer loop for sending commands to the robot
        self.timer_state = self.create_timer(1.0/MASTER_HZ, self.publish_state)

    def start_callback(self, msg):
        if msg.data:
            self.get_logger().info('Received start signal, beginning control loop.')
        else:
            self.get_logger().info('Received stop signal, halting control loop.')
            self.start_collection_pub.publish(Bool(data=False))
        self.start = msg.data
   

    def manual_gripper_callback(self, msg):
        self.current_manual_gripper_cmd = msg

    def manual_servo_callback(self, msg):
        if self.mode == 'manual':
            self.arm.set_servo_cartesian(np.array(msg.data, dtype=np.float32), speed=300, mvacc=2000)

    def auto_gripper_callback(self, msg):
        self.current_auto_gripper_cmd = msg

    def auto_servo_callback(self, msg):
        self.current_auto_servo_cmd = msg
        

    def mode_callback(self, msg):
        self.mode = 'manual' if msg.data else 'auto'
        self.get_logger().info(f'Mode changed to: {self.mode}')

    def publish_state(self):
        servo_state_msg = Float32MultiArray(data=self.arm.get_position()[1])
        gripper_state_msg = Int32(data=self.arm.get_gripper_position()[1])
        self.servo_state_pub.publish(servo_state_msg)
        self.gripper_state_pub.publish(gripper_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MasterController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()