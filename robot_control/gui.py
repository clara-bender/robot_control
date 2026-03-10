import rclpy
from rclpy.node import Node
import tkinter as tk
from threading import Thread
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from PIL import ImageTk
from tkinter import messagebox

# ROS 2 Node class
class ButtonPublisher(Node):
    def __init__(self):
        super().__init__('button_publisher')
        self.start_stop_pub = self.create_publisher(Bool, 'start_button', 10)
        self.failure_pub = self.create_publisher(Bool, 'failure_button', 10)
        self.execution_pub = self.create_publisher(Bool, 'execution_selection',10)
        self.save_pub = self.create_publisher(Bool,'save_button',10)
        self.get_logger().info("Button Publisher Node Started!")
        self.bridge = CvBridge()
        self.current_wrist_image = None
        self.current_tripod_image = None
        self.manual_mode = False

        # Camera Subscriptions
        self.wrist_sub = self.create_subscription(Image, 'camera_image/wrist', self.wrist_camera_callback, 10)
        self.tripod_sub = self.create_subscription(Image, 'camera_image/tripod', self.tripod_camera_callback, 10)
        self.mode_sub = self.create_subscription(Bool,'/manual_mode',self.mode_callback,10)

    def publish_message(self, msg_data, publisher):
        msg = Bool()
        msg.data = msg_data
        publisher.publish(msg)
        self.get_logger().info(f"Publishing: {msg_data}")

    def mode_callback(self,msg):
        self.manual_mode = msg.data

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

        # Failure Button
        self.failure_button = tk.Button(self.window, text="Failure", command=self.failure_button_clicked)
        self.failure_button.pack(pady=10)

        # Choose whether or not to actually execute the commands
        self.execution_var = tk.BooleanVar(value=False)
        self.toggle = tk.Checkbutton(
            self.window,
            text="Robot active",
            variable=self.execution_var
        )
        self.toggle.pack(pady=20)

        # Label to display "correction" or "inference"
        self.correction_label = tk.Label(self.window, text="inference", font=("Arial", 16, "bold"), fg="orange")
        self.correction_label.pack(pady=10)

        # Label to display the image
        self.image_label1 = tk.Label(self.window)
        self.image_label2 = tk.Label(self.window)
        self.image_label1.pack(padx=10, pady=10)
        self.image_label2.pack(padx=10, pady=10)

        # Start a separate thread to update the image and text
        self.update_image_thread = Thread(target=self.update_image)
        self.update_image_thread.daemon = True
        self.update_image_thread.start()

    def start_stop_button_clicked(self):
        current_text = self.start_stop_button.cget("text")
        if current_text == "Start":
            self.ros_node.publish_message(True, self.ros_node.start_stop_pub)
            self.ros_node.publish_message(self.execution_var.get(), self.ros_node.execution_pub)
            self.start_stop_button.config(text="Stop")
        elif current_text == "Stop":
            self.ros_node.publish_message(False, self.ros_node.start_stop_pub)
            self.start_stop_button.config(text="Start")
            answer = messagebox.askyesno(
            "Save Episode",
            "Would you like to save this episode?"
        )
            self.ros_node.publish_message(answer, self.ros_node.save_pub)

    def failure_button_clicked(self):
        self.ros_node.publish_message(True, self.ros_node.failure_pub)


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

            if self.ros_node.manual_mode:
                # If manual_mode is True, display green bold "correction"
                self.correction_label.config(
                    text="correction",  # Text to display
                    fg="green",          # Green color
                    font=("Arial", 16, "bold")  # Bold font with size 16
                )
            else:
                # If manual_mode is False, display orange bold "inference"
                self.correction_label.config(
                    text="inference",  # Text to display
                    fg="orange",       # Orange color
                    font=("Arial", 16, "bold")  # Bold font with size 16
                )

    # def update_text(self):
    #         if self.ros_node.manual_mode:
    #             # If manual_mode is True, display green bold "correction"
    #             self.correction_label.config(
    #                 text="correction",  # Text to display
    #                 fg="green",          # Green color
    #                 font=("Arial", 16, "bold")  # Bold font with size 16
    #             )
    #         else:
    #             # If manual_mode is False, display orange bold "inference"
    #             self.correction_label.config(
    #                 text="inference",  # Text to display
    #                 fg="orange",       # Orange color
    #                 font=("Arial", 16, "bold")  # Bold font with size 16
    #             )


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