import cv2
import numpy as np

points = []
img = None
tattoo = None
original = None

def click_event(event, x, y, flags, param):
    # get 4 pts w mouse click
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Point selected: {x}, {y}")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 points", img)

def order_points(pts):
    # order pts
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")

def resize_tattoo(tattoo, target_width=200):
    # Resize tattoo to a smaller width before warping
    h, w = tattoo.shape[:2]
    scale = target_width / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(tattoo, new_size, interpolation=cv2.INTER_AREA)

def warp_tattoo(tattoo, dst_points, output_shape):
    # warp tattoo img to quadrilateral defined by pts
    h, w = tattoo.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src_points, np.float32(dst_points))
    warped = cv2.warpPerspective(tattoo, M, (output_shape[1], output_shape[0]))
    return warped

def blend_tattoo(photo, warped_tattoo):
    # blend warped RGBA tattoo onto RGB photo
    b, g, r, a = cv2.split(warped_tattoo)
    tattoo_rgb = cv2.merge((b, g, r))
    mask = a / 255.0
    mask_3 = mask[..., None]
    blended = (mask_3 * tattoo_rgb + (1 - mask_3) * photo).astype(np.uint8)
    return blended

# load imgs
img = cv2.imread("arm.jpg")
if img is None:
    raise FileNotFoundError("Could not load arm.jpg")
original = img.copy()

tattoo = cv2.imread("tattoo.png", cv2.IMREAD_UNCHANGED)
if tattoo is None or tattoo.shape[2] != 4:
    raise ValueError("Tattoo must be a PNG with transparency (RGBA)")

# Resize tattoo before use
tattoo = resize_tattoo(tattoo, target_width=200)

while True:
    img = original.copy()
    points = []

    cv2.imshow("Select 4 points", img)
    cv2.setMouseCallback("Select 4 points", click_event)

    # Wait until 4 points are selected
    while len(points) < 4:
        cv2.waitKey(1)

    # Order points for warp
    ordered_points = order_points(points)

    # Draw preview quad
    preview = img.copy()
    cv2.polylines(preview, [np.int32(ordered_points)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Preview Quad", preview)
    print("4 pts selected. Press 'c' to confirm, or 'r' to reset.")

    # Wait for user decision
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # confirm
            cv2.destroyAllWindows()
            warped = warp_tattoo(tattoo, ordered_points, img.shape[:2])
            result = blend_tattoo(original, warped)
            cv2.imshow("Result", result)
            cv2.imwrite("arm_with_curved_tattoo.jpg", result)
            print("Saved as arm_with_curved_tattoo.jpg")
            print("press 0 to close out")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()
        elif key == ord('r'):  # reset
            print("🔄 Resetting selection, please re-click 4 points...")
            cv2.destroyAllWindows()
            break

