import argparse
import numpy as np
import cv2
import os

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    raise ImportError(
        "segment_anything is required. Please install the Segment Anything Model Python package."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (if not provided, will use default for model type)",
    )
    parser.add_argument(
        "--input_prompts",
        type=str,
        default=None,
        help="Path to input prompts CSV (x,y,label per line). If not provided, use center point.",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        help="SAM model type (e.g., vit_h, vit_l, vit_b)",
    )
    parser.add_argument(
        "--output", type=str, default="sam_mask.png", help="Output mask image path"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Determine checkpoint path based on model type if not provided
    default_checkpoints = {
        "vit_h": "/home/justin/code/Manipulator-Software/perception/sam/checkpoints/sam_vit_h_4b8939.pth",
        "vit_l": "/home/justin/code/Manipulator-Software/perception/sam/checkpoints/sam_vit_l_0b3195.pth",
        "vit_b": "/home/justin/code/Manipulator-Software/perception/sam/checkpoints/sam_vit_b_01ec64.pth",
    }
    checkpoint_path = (
        args.sam_checkpoint
        if args.sam_checkpoint
        else default_checkpoints.get(args.sam_model_type)
    )
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found for model type {args.sam_model_type}. Please provide --sam_checkpoint."
        )

    sam = sam_model_registry[args.sam_model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Load prompts from file or use center point
    h, w = image_rgb.shape[:2]
    if args.input_prompts and os.path.exists(args.input_prompts):
        prompts = np.loadtxt(args.input_prompts, delimiter=",")
        if prompts.ndim == 1:
            prompts = prompts[None, :]  # Single prompt
        point_coords = prompts[:, :2]
        point_labels = prompts[:, 2].astype(int)
    else:
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, multimask_output=True
    )
    mask = masks[np.argmax(scores)]

    # Overlay mask on image for visualization
    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)  # Green overlay
    vis = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    cv2.imshow("SAM Segmentation", vis)
    while True:
        # Wait for 100ms for a key event
        key = cv2.waitKey(100)
        # If any key is pressed, break
        if key != -1:
            break
        # If the window was closed, break
        if cv2.getWindowProperty("SAM Segmentation", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    # Save mask
    cv2.imwrite(args.output, (mask * 255).astype(np.uint8))
    print(f"Mask saved to {args.output}")


if __name__ == "__main__":
    main()
