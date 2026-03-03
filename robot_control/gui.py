from tkinter import messagebox

import rclpy
from rclpy.node import Node
import tkinter as tk
from threading import Thread
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from PIL import ImageTk

# ROS 2 Node class
class ButtonPublisher(Node):
    def __init__(self):
        super().__init__('button_publisher')
        self.start_stop_pub = self.create_publisher(Bool, 'start_button', 10)
        self.failure_success_pub = self.create_publisher(Bool, 'failure_success_button', 10)
        self.save_pub = self.create_publisher(Bool, 'save_button', 10)
        self.discard_pub = self.create_publisher(Bool, 'discard_button', 10)
        self.get_logger().info("Button Publisher Node Started!")
        self.bridge = CvBridge()
        self.current_wrist_image = None
        self.current_tripod_image = None

        # Camera Subscriptions
        self.wrist_sub = self.create_subscription(Image, 'camera_image/wrist', self.wrist_camera_callback, 10)
        self.tripod_sub = self.create_subscription(Image, 'camera_image/tripod', self.tripod_camera_callback, 10)

    def publish_message(self, msg_data, publisher):
        msg = Bool()
        msg.data = msg_data
        publisher.publish(msg)
        self.get_logger().info(f"Publishing: {msg_data}")

    def wrist_camera_callback(self, msg):
        # Convert the incoming message to an OpenCV image using CvBridge
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.current_wrist_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting wrist camera image: {e}")

    def tripod_camera_callback(self, msg):
        # Convert the incoming message to an OpenCV image using CvBridge
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.current_tripod_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting tripod camera image: {e}")

# Tkinter GUI class
class MyWindow:
    def __init__(self, ros_node):
        self.ros_node = ros_node  # Reference to the ROS node
        self.window = tk.Tk()
        self.window.title("ROS 2 Button Publisher")

        # Start/Stop Button
        self.start_stop_button = tk.Button(self.window, text="Start", command=self.start_stop_button_clicked)
        self.start_stop_button.pack(pady=10)

        # Failure/Success Button
        self.failure_success_button = tk.Button(self.window, text="Failure", command=self.failure_success_button_clicked)
        self.failure_success_button.pack(pady=10)

        # Label to display the image
        self.image_label1 = tk.Label(self.window)
        self.image_label2 = tk.Label(self.window)
        self.image_label1.pack(padx=10, pady=10)
        self.image_label2.pack(padx=10, pady=10)

        # Start a separate thread to update the image
        self.update_image_thread = Thread(target=self.update_image)
        self.update_image_thread.daemon = True
        self.update_image_thread.start()

    def start_stop_button_clicked(self):
        current_text = self.start_stop_button.cget("text")
        if current_text == "Start":
            self.ros_node.publish_message(True, self.ros_node.start_stop_pub)
            self.start_stop_button.config(text="Stop")
        elif current_text == "Stop":
            self.ros_node.publish_message(False, self.ros_node.start_stop_pub)
            # Ask the user if they want to save the episode before stopping
            response = messagebox.askyesno("Save Episode", "Do you want to save the episode?")
            if response:  # If the user clicks "Yes"
                self.ros_node.publish_message(True, self.ros_node.save_pub)
            else:  # If the user clicks "No"
                self.ros_node.publish_message(True, self.ros_node.discard_pub)
            
            # Change the button text back to "Start"
            self.start_stop_button.config(text="Start")

    def failure_success_button_clicked(self):
        current_text = self.failure_success_button.cget("text")
        if current_text == "Failure":
            self.ros_node.publish_message(True, self.ros_node.failure_success_pub)
            self.failure_success_button.config(text="Success")
        elif current_text == "Success":
            self.ros_node.publish_message(False, self.ros_node.failure_success_pub)
            self.failure_success_button.config(text="Failure")
            self.start_stop_button.config(text="Start")

    def update_image(self):
        """Update the Tkinter image label with new images from ROS 2."""
        while True:
            if self.ros_node.current_wrist_image is not None:
                cv_image = self.ros_node.current_wrist_image
                pil_image = PILImage.fromarray(cv_image)  # Convert OpenCV to PIL
                tk_image = ImageTk.PhotoImage(pil_image)  # Convert PIL to Tkinter format

                # Update the label with the new image
                self.image_label1.config(image=tk_image)
                self.image_label1.image = tk_image  # Keep a reference to avoid garbage collection
            if self.ros_node.current_tripod_image is not None:
                cv_image = self.ros_node.current_tripod_image
                pil_image = PILImage.fromarray(cv_image)  # Convert OpenCV to PIL
                tk_image = ImageTk.PhotoImage(pil_image)  # Convert PIL to Tkinter format

                # Update the label with the new image
                self.image_label2.config(image=tk_image)
                self.image_label2.image = tk_image  # Keep a reference to avoid garbage collection


    def start(self):
        # Start Tkinter's main loop in a separate thread
        self.window.mainloop()

# Main function to initialize ROS 2 node and Tkinter GUI
def main(args=None):
    rclpy.init(args=args)

    # Create the ROS 2 node
    ros_node = ButtonPublisher()

    # Create the Tkinter window
    gui = MyWindow(ros_node)

    # Start ROS 2 node in a separate thread to avoid blocking Tkinter
    def ros_spin():
        rclpy.spin(ros_node)

    ros_thread = Thread(target=ros_spin)
    ros_thread.daemon = True  # Ensures the thread will close when the main program exits
    ros_thread.start()

    # Start the Tkinter GUI in the main thread
    gui.start()

    rclpy.shutdown()

if __name__ == "__main__":
    main()