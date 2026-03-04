import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, Float32MultiArray
from xarm.wrapper import XArmAPI
import time
import numpy as np

class MasterController(Node):
    def __init__(self):
        super().__init__('master_controller')

        self.MASTER_HZ = 20.0

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
        self.get_logger().info('XArm connected and initialized.')

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

        # Publisher: indicate if collection should start (for collection.py)
        self.start_collection_pub = self.create_publisher(Bool, '/start_collection', 10)

        # Publisher: frequency for the master loop
        self.frequency_pub = self.create_publisher(Float32, '/master_frequency', 10)

        # Initializations
        self.mode = 'auto'  # start in auto mode
        self.collecting = False

        

        

    def start_callback(self, msg):
        if msg.data:
            self.arm.set_mode(1)
            self.arm.set_state(0)
            self.get_logger().info('Received start signal, beginning control loop.')
            self.timer_state = self.create_timer(1.0/self.MASTER_HZ, self.publish_state)

            self.frequency_pub.publish(Float32(data=self.MASTER_HZ))
            self.get_logger().info(f'Master Controller initialized, running at {self.MASTER_HZ} Hz')

            
        else:
            self.get_logger().info('Received stop signal, halting control loop.')
            self.timer_state.cancel()
            self.go_home()

    def go_home(self):
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_gripper_enable(enable=True)
            self.arm.set_gripper_mode(0)
            self.arm.set_state(0)
            cmd_joint_pose = [0.0, -90.4, -24.0, 0.0, 61.3, 180.0] 
            cmd_gripper_pose = 850.0
            self.arm.set_servo_angle(servo_id=8, angle=cmd_joint_pose, is_radian=False, wait=True) 
            self.arm.set_gripper_position(cmd_gripper_pose, wait=True)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(4) # clears cached commands and stops the arm
   
    def manual_gripper_callback(self, msg):
        if self.mode == 'manual':
            self.arm.set_gripper_position(int(msg.data))
            if not self.collecting:
                self.start_collection_pub.publish(Bool(data=True))
                self.collecting = True

    def manual_servo_callback(self, msg):
        if self.mode == 'manual':
            self.arm.set_servo_cartesian(np.array(msg.data, dtype=np.float32), speed=300, mvacc=2000)
            if not self.collecting:
                self.start_collection_pub.publish(Bool(data=True))
                self.collecting = True

    def auto_gripper_callback(self, msg):
        if self.mode == 'auto':
            self.arm.set_gripper_position(int(msg.data))
            if not self.collecting:
                self.start_collection_pub.publish(Bool(data=True))
                self.collecting = True

    def auto_servo_callback(self, msg):
        if self.mode == 'auto':
            self.arm.set_servo_cartesian(np.array(msg.data, dtype=np.float32), speed=300, mvacc=2000)
            if not self.collecting:
                self.start_collection_pub.publish(Bool(data=True))
                self.collecting = True

    def mode_callback(self, msg):
        self.mode = 'manual' if msg.data else 'auto'
        self.get_logger().info(f'Mode changed to: {self.mode}')

    def publish_state(self):
        servo_state_msg = Float32MultiArray(data=self.arm.get_position()[1])
        gripper_state_msg = Int32(data=self.arm.get_gripper_position()[1])
        self.servo_state_pub.publish(servo_state_msg)
        self.gripper_state_pub.publish(gripper_state_msg)
        self.get_logger().info(f'Published servo state: {servo_state_msg.data}, gripper state: {gripper_state_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = MasterController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()