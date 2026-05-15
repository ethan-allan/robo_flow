# Hardware Cfgs

Design of this package

- Sensor cfgs store the information describing how to connect, run, process the sensor data and what format it should be in. This should be fully encapsulating the sensor such that no additional information is required by the client
-

Env_cfg 
    - ur3_pos_force_vis1

    Observation cfg
        - realsense_hand.yaml
        - realsense_platform.yaml
        - ur3_force
        - ur3_tcp
        - gelsight

    Action cfg
        - ur3_joint
        - ur3_tcp_pos
        - ur5_tcp_pos

