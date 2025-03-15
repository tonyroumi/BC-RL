# BC-RL : Behavior Cloning with Reinforcement Learning using Transformer-Based Sensor Fusion for Autonomous Driving


This work processes and fuses information from multi-modal multi-view sensors as input to a hybrid learning framework for autonomous driving. 

# Contents
- [Setup](#setup)
- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

# Setup
**Clone the repoistory and build the environment**
```
git clone https://github.com/BC-RL/BC-RL.git
cd BC-RL
conda create -n BC-RL python=3.7
conda activate BC-RL
pip3 install -r requirements.txt
```

**Download and setup CARLA 0.9.10.1**
```
chmod +x setup_carla.sh
./setup_carla.sh
```


# Data Generation
> **Note:** The data generation methodology and scripts are adapted from [Interfuser](https://github.com/opendilab/InterFuser).

The collected dataset is structured as follows:
```
- TownX_{tiny,short,long}: corresponding to different towns and routes files
    - routes_X: contains data for an individual route
        - rgb_{front, left, right, rear}: multi-view camera images at 400x300 resolution
        - seg_{front, left, right}: corresponding segmentation images
        - depth_{front, left, right}: corresponding depth images
        - lidar: 3d point cloud in .npy format
        - birdview: topdown segmentation images required for training LBC
        - 2d_bbs_{front, left, right, rear}: 2d bounding boxes for different agents in the corresponding camera view
        - 3d_bbs: 3d bounding boxes for different agents
        - affordances: different types of affordances
        - measurements: contains ego-agent's position, velocity and other metadata
        - other_actors: contains the positions, velocities and other metadatas of surrounding vehicles and the traffic lights
```

### 1.) Generate scripts for collecting data in batches
```
cd dataset
python init_dir.py
cd ..
cd data_collection
python generate_yamls.py # You can modify fps, waypoints distribution strength ...

# If you don't need all weather, you can modify the following script
python generate_bashs.py
python generate_batch_collect.py 
cd ..
```

### 2.) Run CARLA Servers
```
# start 14 carla servers: ip [localhost], port [20000 - 20026]
cd carla
CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh --world-port=20000 -opengl &
CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh --world-port=20002 -opengl &
...
CUDA_VISIBLE_DEVICES=7 ./CarlaUE4.sh --world-port=20026 -opengl &
```
### 3.) Generate Data
Run batch scripts of the town and route type that you need to collect. (This will contain all 14 kinds of weather.)

**NOTE:** If you don't need all weather types you can modify `generate_bashs.py`.
```
bash data_collection/batch_run/run_route_routes_town01_long.sh
bash data_collection/batch_run/run_route_routes_town01_short.sh
...
bash data_collection/batch_run/run_route_routes_town07_tiny.sh
```





# Acknowledgements
This implementation is based on code from several repositories:
- [Transfuser](https://github.com/autonomousvision/transfuser)
- [Interfuser](https://github.com/opendilab/InterFuser)
- [Roach](https://github.com/zhejz/carla-roach)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)


