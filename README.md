# NUSHigh 2026
Sample code for UGOT control.

Each file in the `src` folder contains a program demonstrating a different function.
For UGOT files, the function names indicate the type of UGOT used in the program as follows:
- SB: self-balancing vehicle
- WL: wheel-legged robot
- mec: mecanum wheel car

For example, `UGOT_line_follow.py` contains code for any UGOT robot to follow a line.
Within `UGOT_line_follow.py`, the function `line_follow_SB()` is used specifically for the self-balancing car.

You should upload or create the `constants.py` file in UPython before using any of the main program files. Change the `ROBOT_TYPE` and `IP_ADDRESS` variables according to the robot you are using.

Note that since the base of the engineering robot is the mecanum wheel car, there is no separate code for the engineering robot chassis movement.

If there is no robot specified in the name, the code is valid for all types of robots, or does not need a robot.

## Getting started
1. Visit the UPython [website](py.ubtrobot.com/gl). 
2. Change the language using the settings at the bottom left corner of the screen.![](language.png)
3. (Optional) Create an account using your personal (not school) email or phone number to save your code to the cloud. If you do not create an account, make sure to regularly export your workspace to a `.upy` file.
4. Create a new `.ipynb` file and paste the following command in a cell: `pip install ugot opencv-python numpy fer`. Click the triangle next to the cell to execute. You only need to do this once per computer.![alt text](install.png)
5. Upload or create your Python files in the file explorer on the left.

Happy coding!