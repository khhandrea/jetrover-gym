import manipulation.utils.visualization_util as vis_util
from manipulation.kinematics import inverse_kinematics
import argparse

if __name__ == "__main__":
    vis_util.run_urdf_viewer_with_ik(print_error=True, ik_func=inverse_kinematics)