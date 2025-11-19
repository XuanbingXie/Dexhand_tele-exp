import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Robotic_Arm.rm_robot_interface import *

# RM65 we use
arm_models_to_points = {  
    "RM_65": [  
        [0, 20, 70, 0, 90, 0],
        [0.3, 0, 0.3, 3.14, 0, 0],
        [0.2, 0, 0.3, 3.14, 0, 0],
        [0.3, 0, 0.3, 3.14, 0, 0],
        [0.2, 0.05, 0.3, 3.14, 0, 0],
        [0.2, -0.05, 0.3, 3.14, 0, 0] ,
    ]
} 


class RobotArmController:
    def __init__(self, ip, port, level=3, mode=2):
        """
        Initialize and connect to the robotic arm.

        Args:
            ip (str): IP address of the robot arm.
            port (int): Port number.
            level (int, optional): Connection level. Defaults to 3.
            mode (int, optional): Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
        """
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            print("\nFailed to connect to the robot arm\n")
            exit(1)
        else:
            print(f"\nSuccessfully connected to the robot arm: {self.handle.id}\n")

    def get_arm_model(self):
        """Get robotic arm mode.
        """
        res, model = self.robot.rm_get_robot_info()
        if res == 0:
            return model["arm_model"]
        else:
            print("\nFailed to get robot arm model\n")

    def disconnect(self):
        """
        Disconnect from the robot arm.

        Returns:
            None
        """
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            print("\nSuccessfully disconnected from the robot arm\n")
        else:
            print("\nFailed to disconnect from the robot arm\n")

    def get_arm_software_info(self):
        """
        Get the software information of the robotic arm.

        Returns:
            None
        """
        software_info = self.robot.rm_get_arm_software_info()
        if software_info[0] == 0:
            print("\n================== Arm Software Information ==================")
            print("Arm Model: ", software_info[1]['product_version'])
            print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
            print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
            print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
            print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
            print("==============================================================\n")
        else:
            print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")

    def get_current_pose(self):
        """
        获取机械臂当前末端位姿 [x, y, z, rx, ry, rz]
        """
        joint_angles = self.robot.rm_get_joint_degree()[1]
        pose = self.robot.rm_algo_forward_kinematics(joint_angles)
        return pose

    def movej(self, joint, v=20, r=0, connect=0, block=1):
        """
        Perform movej motion.

        Args:
            joint (list of float): Joint positions.
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movej_result = self.robot.rm_movej(joint, v, r, connect, block)
        if movej_result == 0:
            print("\nmovej motion succeeded\n")
        else:
            print("\nmovej motion failed, Error code: ", movej_result, "\n")

    def movel(self, pose, v=20, r=0, connect=0, block=1):
        """
        Perform movel motion.

        Args:
            pose (list of float): End position [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movel_result = self.robot.rm_movel(pose, v, r, connect, block)
        if movel_result == 0:
            print("\nmovel motion succeeded\n")
        else:
            print("\nmovel motion failed, Error code: ", movel_result, "\n")

    def movec(self, pose_via, pose_to, v=20, r=0, loop=0, connect=0, block=1):
        """
        Perform movec motion.

        Args:
            pose_via (list of float): Via position [x, y, z, rx, ry, rz].
            pose_to (list of float): End position for the circular path [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            loop (int, optional): Number of loops. Defaults to 0.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movec_result = self.robot.rm_movec(pose_via, pose_to, v, r, loop, connect, block)
        if movec_result == 0:
            print("\nmovec motion succeeded\n")
        else:
            print("\nmovec motion failed, Error code: ", movec_result, "\n")

    def movej_p(self, pose, v=20, r=0, connect=0, block=1):
        """
        Perform movej_p motion.

        Args:
            pose (list of float): Position [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("\nmovej_p motion succeeded\n")
        else:
            print("\nmovej_p motion failed, Error code: ", movej_p_result, "\n")

    def move_relative(self, delta_pose, v=20):
        """
        基于当前末端位姿做相对移动
        delta_pose: [dx, dy, dz, drx, dry, drz]
        """
        current_pose = self.get_current_pose()
        target_pose = [c + d for c, d in zip(current_pose, delta_pose)]
        self.movel(target_pose, v=v)

def main():
    # Create a robot arm controller instance and connect to the robot arm
    robot_controller = RobotArmController("192.168.1.28", 8080, 3)

    arm_model = robot_controller.get_arm_model()
    points = arm_models_to_points.get(arm_model, [])

    # init point

    # point = [0.468946,-0.142716,-0.019291,-2.955,-0.964,0.607]
    point = [0.5975356110297744, -0.08434494306916175, -0.11, -2.8520692557450005, 0.8996311980742325, -2.0166478615686576]
    
    robot_controller.movel(point, v = 5)
    pose = robot_controller.get_current_pose()
    print("当前末端位姿:", pose)
    
    # robot_controller.move_relative([0, 0, 0, 0, 0, 0], v=10)

    # Perform movej_p motion
    # robot_controller.movej_p(points[1])

    # point1 = [0.5666403461081977, 0.016707402418104866, -0.10, 2.999640464782715, -1.1000466346740723, 0.9003937840461731]
    # point1 = [0.5862669430455809, 0.011726378869284265, -0.14, 2.99984073638916, -1.1000133752822876, 0.9000556468963624]
    # robot_controller.movel(point1, v = 20)

    # # Perform movej_p motion again
    # robot_controller.movej_p(points[3])

    # # Perform movec motion
    # robot_controller.movec(points[4], points[5], loop=2)

    # Disconnect the robot arm
    robot_controller.disconnect()


if __name__ == "__main__":
    main()
