import os
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError(
        "cv2 (OpenCV) is required. Please install it with 'pip install opencv-python'."
    )

try:
    import lcm
except ImportError:
    raise ImportError(
        "lcm is required. Please install the LCM Python bindings in your environment."
    )

import threading
import time

try:
    import trimesh
except ImportError:
    raise ImportError(
        "trimesh is required. Please install it with 'pip install trimesh'."
    )

import imageio
import logging

# Import your FoundationPose and related utilities
from estimater import *
from datareader import *

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    raise ImportError(
        "segment_anything is required. Please install the Segment Anything Model Python package."
    )

# --- LCM message types ---
# You may need to adapt these imports to your LCM message definitions
# from your_lcm_types import rgb_msg_t, depth_msg_t


class RGBDLCMSubscriber:
    def __init__(self, rgb_channel, depth_channel):
        self.lc = lcm.LCM()
        self.rgb_channel = rgb_channel
        self.depth_channel = depth_channel
        self.rgb = None
        self.depth = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.start()

    def _on_rgb(self, channel, data):
        # Replace with your actual LCM RGB message parsing
        # msg = rgb_msg_t.decode(data)
        # self.rgb = msg.data.reshape((msg.height, msg.width, 3))
        pass

    def _on_depth(self, channel, data):
        # Replace with your actual LCM Depth message parsing
        # msg = depth_msg_t.decode(data)
        # self.depth = msg.data.reshape((msg.height, msg.width))
        pass

    def _listen(self):
        self.lc.subscribe(self.rgb_channel, self._on_rgb)
        self.lc.subscribe(self.depth_channel, self._on_depth)
        while self.running:
            self.lc.handle_timeout(100)

    def get_latest(self):
        with self.lock:
            return self.rgb, self.depth

    def stop(self):
        self.running = False
        self.thread.join()


def run_sam1_on_image(sam_predictor, image):
    # Use SAM1 to segment the object in the first frame
    # This is a placeholder: you may want to use a prompt or bounding box
    # For demo, use the center of the image as a point prompt
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    masks, scores, logits = sam_predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )
    # Use the highest scoring mask
    best_mask = masks[np.argmax(scores)]
    return best_mask.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument(
        "--K",
        type=str,
        required=True,
        help="Camera intrinsics as comma-separated 9 values",
    )
    parser.add_argument("--rgb_channel", type=str, default="CAMERA_RGB")
    parser.add_argument("--depth_channel", type=str, default="CAMERA_DEPTH")
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--debug_dir", type=str, default="./debug_lcm")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.debug_dir, exist_ok=True)
    set_logging_format()
    set_seed(0)

    # Load mesh
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # Parse camera intrinsics
    K = np.array([float(x) for x in args.K.split(",")]).reshape(3, 3)

    # Initialize FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    # Initialize SAM1
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam_predictor = SamPredictor(sam)

    # Start LCM subscriber
    lcm_sub = RGBDLCMSubscriber(args.rgb_channel, args.depth_channel)

    try:
        initialized = False
        pose = None
        frame_idx = 0
        while True:
            rgb, depth = lcm_sub.get_latest()
            if rgb is None or depth is None:
                time.sleep(0.01)
                continue

            if not initialized:
                # Run SAM1 on the first RGB frame
                mask = run_sam1_on_image(sam_predictor, rgb)
                pose = est.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.est_refine_iter,
                )
                initialized = True
                logging.info("Initial registration done.")
            else:
                pose = est.track_one(
                    rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter
                )

            # Save or visualize results
            pose_to_save = pose.reshape(4, 4) if pose.shape == (4, 4) else pose
            np.savetxt(f"{args.debug_dir}/pose_{frame_idx:06d}.txt", pose_to_save)
            if args.debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(
                    rgb,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=K,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )
                cv2.imshow("Tracking", vis[..., ::-1])
                cv2.waitKey(1)
            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        lcm_sub.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
