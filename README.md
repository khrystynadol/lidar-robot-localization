### Lidar robot localization

You are given a log of sensor data and commands from a robot driving around a room. You need to determine the coordinates of the robot at the end of the movement and the average speed of the robot. The robot moves only on the floor (2D space). The robot moves by changing its coordinates and does not change its orientation in space. The robot has one sensor - a 2D lidar. The lidar scans the space around the robot. One scan is performed as follows. The lidar emits K rays, the angle between adjacent rays is 360/K degrees, and for each ray determines the distance to the wall in that direction. The lidar has an error Sl in distance (the distance determined by the lidar is assumed to be a random variable from a normal distribution with a standard deviation Sl), and has no error in angle. The data from the lidar comes in ascending order of the angle in an anti-clockwise direction. The first measurement is in the direction of the oX axis. Scanning is instantaneous.

One movement of the robot is its displacement by a vector (X, Y) with a standard deviation along each of the axes (Sx, Sy). The movements along the X-axis and the Y-axis are independent of each other. The robot's life cycle consists of alternate scans and movements.

The room is defined as a polygon without self-intersections

Sample tests are in the `sample_input` folder.
