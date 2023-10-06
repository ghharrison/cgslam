# cgslam
Cooperative Graph-Based SLAM, implemented in Python. This was a term project for AER1515H: Perception for Robotics, taught by Professor Steven Waslander. 

The objective of this project was to make a form of graph-based SLAM that could take advantage of cooperation between multiple mapping robots to obtain a shared map with lower landmark and pose error than the robots could obtain individually. This is done by having the robots achieve a consensus on their relative positions when simultaneously observing each other, translating their maps into a common global frame, and then re-optimizing the now combined maps. 

The core graph functionality of this project comes from Jeff Irion's [python-graphslam](https://github.com/JeffLIrion/python-graphslam), while datasets for the robots used in this simulation come from the University of Toronto UTIAS [Multi-Robot Collaborative Localization and Mapping (MR. CLAM) datasets](http://asrl.utias.utoronto.ca/datasets/mrclam/index.html). The SLAM functionality, data utilities, cooperative map-merging, and simulation were custom-written for this project. Original contributions to this project are present in the files `slam.py` and `slam_utils.py`. 

# References
K. Y. Leung, Y. Halpern, T. D. Barfoot, and H. H. Liu, The UTIAS multi-robot cooperative localization and mapping dataset, Intl. Journal of Robotics Research, vol. 30, no. 8, pp. 969-9
