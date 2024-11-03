import cv2
import numpy as np
from praxis.Camera import Camera


class TrainingDataGenerator:

    def __init__(self):
        self.camera = Camera()

    def generate(self, input_image_file, output_image_file, overlay_file, markers_corners):
        # Load the image you want to overlay (replace 'overlay.jpg' with your image path)
        overlay_image = cv2.imread(overlay_file)
        overlay_height, overlay_width = overlay_image.shape[:2]

        # Define the 3D coordinates of the marker's four corners (replace with your actual coordinates)
        # Example format: [x, y, z]
        marker_3d_coords = np.array([
            markers_corners[0],  # Top-left corner
            markers_corners[1],  # Top-right corner
            markers_corners[2],  # Bottom-right corner
            markers_corners[3]  # Bottom-left corner
        ], dtype=np.float32)

        # Project 3D points onto the 2D image using camera matrix
        # Replace with actual values for camera matrix and distortion coefficients
        camera_matrix = np.array([[self.camera.f_x, 0, cx], [0, self.camera.f_y, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Project 3D marker corners to 2D
        marker_2d_coords, _ = \
            cv2.projectPoints(marker_3d_coords, self.camera.rvec, self.camera.tvec, camera_matrix, dist_coeffs)
        marker_2d_coords = marker_2d_coords.reshape(-1, 2)  # Reshape to a 2D array

        # Define the 2D coordinates for the overlay image (corners of the overlay image)
        image_corners = np.array([
            [0, 0],
            [overlay_width - 1, 0],
            [overlay_width - 1, overlay_height - 1],
            [0, overlay_height - 1]
        ], dtype=np.float32)

        # Compute the homography matrix
        homography_matrix, _ = cv2.findHomography(image_corners, marker_2d_coords)

        # Warp the overlay image onto the marker area in the original image
        output_image = cv2.warpPerspective(overlay_image, homography_matrix, (output_width, output_height))

        # Combine with the original image if needed
        # Assuming original image where marker is present
        final_image = original_image.copy()
        mask = (output_image > 0).all(axis=2)
        final_image[mask] = output_image[mask]

        # Display the result
        cv2.imshow('Replaced Marker', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



