import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


# These lambda functions define how to find the 'hand' (blue) and 'target' (red)
# based on which color channel is dominant in a given pixel.
HAND_CONDITION = lambda frame: (frame[:, :, 2] > frame[:, :, 0]) & (frame[:, :, 2] > frame[:, :, 1]) & (frame[:, :, 2] > 0.1)
TARGET_CONDITION = lambda frame: (frame[:, :, 0] > frame[:, :, 1]) & (frame[:, :, 0] > frame[:, :, 2]) & (frame[:, :, 0] > 0.1)

# --- Data Structure for Tracked Objects ---
@dataclass
class TrackedObject:
    """
    A data structure to hold all information about an object being tracked.
    Using @dataclass automatically generates helpful methods like __init__ and __repr__.
    
    Attributes:
        name (str): A unique identifier for the object (e.g., 'hand').
        color_condition (Callable): A function that returns a boolean mask for the object's pixels.
        highlight_color (str): The color of the highlight border on the plot.
        patch (Optional[patches.Rectangle]): The matplotlib patch object used for highlighting.
                                             It's excluded from the constructor ('init=False')
                                             and the string representation ('repr=False').
        patch_index (Optional[int]): The calculated flattened index of the patch where the
                                     object is located. This is the key piece of information
                                     for the Vision Transformer (ViT) model.
    """
    name: str
    color_condition: Callable[[np.ndarray], np.ndarray]
    highlight_color: str
    patch: Optional[patches.Rectangle] = field(default=None, init=False, repr=False)
    patch_index: Optional[int] = field(default=None, init=False, repr=True)
    # This will store the last valid index, to be used as a fallback.
    last_known_patch_index: Optional[int] = field(default=None, init=False, repr=False)

# --- Image Processing Helper ---
def find_object_center(frame: np.ndarray, color_condition: Callable[[np.ndarray], np.ndarray]) -> Optional[Tuple[float, float]]:
    """
    Finds the geometric center (mean) of pixels that match a given color condition.
    
    Args:
        frame (np.ndarray): The image frame to search within.
        color_condition (Callable): A function that takes the frame and returns a boolean
                                    mask of pixels matching the desired color.
        
    Returns:
        Optional[Tuple[float, float]]: A tuple (center_x, center_y) of the object's
                                        center in pixel coordinates, or None if no
                                        matching pixels are found.
    """
    # np.where returns the indices of elements that are non-zero.
    # For a boolean mask, this gives us the coordinates of all 'True' pixels.
    pixels = np.where(color_condition(frame))
    if pixels[0].size > 0:
        # The result of np.where is a tuple of arrays, one for each dimension.
        # For a 2D mask, it's (array_of_y_coords, array_of_x_coords).
        y_coords, x_coords = pixels
        # We calculate the mean of the coordinates to find the center of the object.
        center_y, center_x = np.mean(y_coords), np.mean(x_coords)
        return center_x, center_y
    # If no pixels match the condition, the object is not in the frame.
    return None

# --- Main Plotting Class ---
class InteractivePlot:
    """
    Manages an interactive plot to visualize a grid-based environment,
    highlight tracked objects, and calculate their patch indices for a ViT.
    """
    def __init__(self, image_size: int, patch_size: int):
        """
        Initializes the plot and the patch configuration.
        
        Args:
            image_size (int): The total size (width and height) of the input image in pixels.
            patch_size (int): The size (width and height) of a single square patch.
        """
        self.image_size = image_size
        self.patch_size = patch_size  # This is the 'cell_size' for grid visualization
        self.tracked_objects = {}

        # --- Patch Configuration for ViT ---
        # The number of patches along the width (or height) of the image.
        # This is crucial for calculating the flattened index later.
        # Example: A 32x32 image with 4x4 patches has 32 // 4 = 8 patches per row.
        self.num_patches_per_row = image_size // patch_size

        # --- Matplotlib Setup ---
        plt.ion()  # Enable interactive mode for live plot updates.
        self.fig, self.ax = plt.subplots()
        # Create a placeholder for the image. It will be updated with new frames.
        self.img_display = self.ax.imshow(np.zeros((image_size, image_size, 3)))

        # --- Grid and Plot Styling ---
        self._setup_grid()
        self._cleanup_plot()

    def _setup_grid(self):
        """Draws a grid on the plot to visualize the patches."""
        # Calculate the positions for the grid lines. They are placed at the edge of each patch.
        tick_positions = np.arange(self.patch_size, self.image_size, self.patch_size) - 0.5
        self.ax.set_xticks(tick_positions)
        self.ax.set_yticks(tick_positions)
        self.ax.grid(which='major', color='black', linestyle='-', linewidth=1)

    def _cleanup_plot(self):
        """Removes distracting axis labels and ticks for a clean visualization."""
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(axis='both', which='both', length=0)

    def add_tracked_object(self, name: str, color_condition: Callable[[np.ndarray], np.ndarray], highlight_color: str):
        """
        Registers a new object to be tracked.
        
        Args:
            name (str): The unique name for the object (e.g., 'hand', 'target').
            color_condition (Callable): A function defining how to find the object.
            highlight_color (str): The color for the highlight rectangle.
        """
        if name in self.tracked_objects:
            print(f"Warning: Tracked object '{name}' already exists and will be overwritten.")
        self.tracked_objects[name] = TrackedObject(name, color_condition, highlight_color)

    def _get_patch_indices_from_center(self, center_x: float, center_y: float) -> Tuple[int, int]:
        """
        Converts pixel coordinates of an object's center into 2D patch indices (row, col).
        
        Args:
            center_x (float): The x-coordinate of the object's center in pixels.
            center_y (float): The y-coordinate of the object's center in pixels.
            
        Returns:
            Tuple[int, int]: A tuple (patch_row, patch_col) representing the 2D
                             location of the patch in the grid.
        """
        # Floor division by patch size gives the 0-based index of the patch.
        patch_col = int(center_x // self.patch_size)
        patch_row = int(center_y // self.patch_size)
        return patch_row, patch_col

    def update(self, frame: np.ndarray):
        """
        Updates the plot with a new frame, finds objects, calculates their patch
        indices, and draws highlights.
        
        Args:
            frame (np.ndarray): The new image frame to display.
        """
        # Update the image data on the plot.
        self.img_display.set_data(frame)
        
        for obj in self.tracked_objects.values():
            # --- Reset Visual State ---
            # Remove the old highlight rectangle from the plot if it exists.
            if obj.patch:
                obj.patch.remove()
            obj.patch = None
            # The patch_index will be determined below.
            obj.patch_index = None

            # --- Find Object and Calculate Index ---
            center = find_object_center(frame, obj.color_condition)
            
            linestyle = 'solid'  # Default to solid line for active detection
            
            if center is not None:
                # --- CASE 1: OBJECT IS FOUND ---
                center_x, center_y = center
                # 1. Convert pixel center to 2D patch grid coordinates.
                patch_row, patch_col = self._get_patch_indices_from_center(center_x, center_y)
                
                # 2. Calculate the flattened 1D patch index.
                current_index = patch_row * self.num_patches_per_row + patch_col
                
                # 3. Update both the current index and the last known index.
                obj.patch_index = current_index
                obj.last_known_patch_index = current_index
            
            else:
                # --- CASE 2: OBJECT IS NOT FOUND ---
                # Use the last known position as a fallback.
                obj.patch_index = obj.last_known_patch_index
                # Use a dashed line to indicate this is a "ghost" of the last known position.
                linestyle = 'dashed'

            # --- Draw Highlight (if a position is known) ---
            # This block now runs whether the object was found or not, as long
            # as there is a last known position to fall back on.
            if obj.patch_index is not None:
                # Re-calculate row and col from the (potentially fallback) index.
                patch_row = obj.patch_index // self.num_patches_per_row
                patch_col = obj.patch_index % self.num_patches_per_row
                
                # Calculate the rectangle's corner for drawing.
                rect_x = patch_col * self.patch_size - 0.5
                rect_y = patch_row * self.patch_size - 0.5
                
                # Create the new rectangle patch with the appropriate line style.
                new_rect = patches.Rectangle(
                    (rect_x, rect_y), self.patch_size, self.patch_size,
                    linewidth=2, edgecolor=obj.highlight_color, facecolor='none', linestyle=linestyle
                )
                self.ax.add_patch(new_rect)
                obj.patch = new_rect
                
        # --- Redraw the Canvas ---
        # After all updates, redraw the plot to show the new frame and highlights.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_patch_index(self, name: str) -> Optional[int]:
        """
        Public method to safely retrieve the calculated patch index for a given object.
        
        Args:
            name (str): The name of the tracked object.
            
        Returns:
            Optional[int]: The flattened patch index if the object was found in the last
                           update, otherwise None.
        """
        if name in self.tracked_objects:
            return self.tracked_objects[name].patch_index
        print(f"Warning: No tracked object with name '{name}' found.")
        return None

    def close(self):
        """Closes the interactive plot window properly."""
        plt.ioff() # Turn off interactive mode.
        plt.show() # Show the final plot until the user closes it.

# --- Example Usage ---
if __name__ == '__main__':
    """
    This block serves as a demonstration and a test for the InteractivePlot class.
    You can run this script directly (`python interactive_plot.py`) to see it in action.
    """
    # --- Configuration ---
    IMAGE_SIZE = 32  # A 32x32 pixel image.
    PATCH_SIZE = 4   # Using 4x4 patches.
    # This configuration results in an 8x8 grid of patches (32/4 = 8).
    
    # --- Create a Plotter ---
    plotter = InteractivePlot(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
    
    # --- Define Color Conditions ---
    # Condition to find blue objects (for the 'hand').
    hand_condition = lambda frame: (frame[:, :, 2] > 0.8) & (frame[:, :, 0] < 0.2)
    # Condition to find red objects (for the 'target').
    target_condition = lambda frame: (frame[:, :, 0] > 0.8) & (frame[:, :, 2] < 0.2)
    
    # --- Register Objects to Track ---
    plotter.add_tracked_object(name='hand', color_condition=hand_condition, highlight_color='g')
    plotter.add_tracked_object(name='target', color_condition=target_condition, highlight_color='r')

    # --- Simulation ---
    try:
        # Create a black background frame.
        frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        
        # --- Place objects in specific patches to test index calculation ---
        # Place hand (blue) in patch at (row=2, col=3).
        # Expected index = 2 * 8 + 3 = 19.
        frame[2*PATCH_SIZE : 3*PATCH_SIZE, 3*PATCH_SIZE : 4*PATCH_SIZE, 2] = 0.9
        
        # Place target (red) in patch at (row=5, col=1).
        # Expected index = 5 * 8 + 1 = 41.
        frame[5*PATCH_SIZE : 6*PATCH_SIZE, 1*PATCH_SIZE : 2*PATCH_SIZE, 0] = 0.9
        
        print("--- Initial State ---")
        plotter.update(frame)
        
        # --- Retrieve and Print Patch Indices ---
        hand_idx = plotter.get_patch_index('hand')
        target_idx = plotter.get_patch_index('target')
        print(f"Hand found at patch index: {hand_idx} (Expected: 19)")
        print(f"Target found at patch index: {target_idx} (Expected: 41)")
        
        plt.pause(3) # Pause to observe the initial state.
        
        # --- Simulate Movement ---
        print("\n--- Simulating Hand Movement ---")
        # Erase old hand position.
        frame[2*PATCH_SIZE : 3*PATCH_SIZE, 3*PATCH_SIZE : 4*PATCH_SIZE, 2] = 0.0
        # Place hand in a new patch at (row=3, col=4).
        # Expected new index = 3 * 8 + 4 = 28.
        frame[3*PATCH_SIZE : 4*PATCH_SIZE, 4*PATCH_SIZE : 5*PATCH_SIZE, 2] = 0.9
        
        plotter.update(frame)
        hand_idx = plotter.get_patch_index('hand')
        print(f"Hand moved to patch index: {hand_idx} (Expected: 28)")
        
        plt.pause(3) # Pause to observe the final state.

    except KeyboardInterrupt:
        print("\nPlotting stopped by user.")
    finally:
        # Ensure the plot is closed cleanly.
        print("Closing plot.")
        plotter.close()