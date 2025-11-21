from LinkerHand.linker_hand_api import LinkerHandApi
def main():
    # 初始化API hand_type:left or right   hand_joint:L7 or L10 or L20 or L25
    linker_hand = LinkerHandApi(hand_type="right", hand_joint="L10")
    # 设置手指速度
    linker_hand.set_speed(speed=[120,200,200,200,200])
    # 设置手扭矩
    linker_hand.set_torque(torque=[200,200,200,200,200])

    #大拇指弯曲#大拇指侧摆#食指弯曲#中指弯曲#无名指弯曲#小指弯曲#食指侧摆#无名指侧摆#小指侧摆#拇指旋转
    # pose = [255, 255, 255, 255, 255, 255, 0, 0, 0, 0]
    # linker_hand.finger_move(pose=pose)
    # pose = [186, 142, 135, 129, 135, 255, 107, 26, 0, 113]
    # linker_hand.finger_move(pose=pose)

    # force
    torque = linker_hand.get_torque()
    print("torque=", torque)
    tem = linker_hand.get_temperature()
    print("tem=",tem)
    speed = linker_hand.get_speed()
    print("speed=",speed)
    joint_speed = linker_hand.get_joint_speed()
    print("joint_speed=",joint_speed)
    force = linker_hand.get_force()
    print("force=",force)
    touch = linker_hand.get_touch()
    print ("touch=",touch)
    touch_type = linker_hand.get_touch_type()
    print ("touch_type=",touch_type)
    state = linker_hand.get_state()
    print ("status=",state)

    

if __name__ == "__main__":
    main()