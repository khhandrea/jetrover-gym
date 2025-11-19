import manipulation.utils.visualization_util as vis_util
from manipulation.kinematics import forward_kinematics
import argparse

if __name__ == "__main__":
    vis_util.run_urdf_viewer_with_fk(print_error=True, fk_func=forward_kinematics)