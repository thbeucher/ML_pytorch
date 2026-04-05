import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

# --- Data Structure for Tracked Objects ---
@dataclass
class TrackedObject:
    """A data structure to hold information about an object to be tracked and highlighted."""
    name: str
    color_condition: Callable[[np.ndarray], np.ndarray]
    highlight_color: str
    patch: Optional[patches.Rectangle] = field(default=None, init=False, repr=False)

# --- Image Processing Helper ---
def find_object_center(frame: np.ndarray, color_condition: Callable[[np.ndarray], np.ndarray]) -> Optional[Tuple[float, float]]:
    """
    Finds the center of an object in a frame based on a color condition.
    
    Args:
        frame (np.ndarray): The current image frame.
        color_condition (Callable): A function that takes a frame and returns a boolean mask for the object's pixels.
        
    Returns:
        Optional[Tuple[float, float]]: A tuple (center_x, center_y) of the object's center, or None if not found.
    """
    pixels = np.where(color_condition(frame))
    if pixels[0].size > 0:
        y_coords, x_coords = pixels
        center_y, center_x = np.mean(y_coords), np.mean(x_coords)
        return center_x, center_y
    return None

# --- Main Plotting Class ---
class InteractivePlot:
    """
    A class to create and manage an interactive matplotlib plot for visualizing a grid-based environment.
    It can display frames and highlight specific cells for any number of tracked objects.
    """
    def __init__(self, image_size: int, cell_size: int):
        """
        Initializes the interactive plot.
        
        Args:
            image_size (int): The size of the image (width and height) in pixels.
            cell_size (int): The size of each grid cell in pixels.
        """
        self.image_size = image_size
        self.cell_size = cell_size
        self.tracked_objects = {}

        # --- Matplotlib Setup ---
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()

        # Create a placeholder for the image display
        self.img_display = self.ax.imshow(np.zeros((image_size, image_size, 3)))

        # --- Grid and Plot Cleanup ---
        self._setup_grid()
        self._cleanup_plot()

    def _setup_grid(self):
        """Sets up the grid lines for the plot."""
        tick_positions = np.arange(self.cell_size, self.image_size, self.cell_size) - 0.5
        self.ax.set_xticks(tick_positions)
        self.ax.set_yticks(tick_positions)
        self.ax.grid(which='major', color='black', linestyle='-', linewidth=1)

    def _cleanup_plot(self):
        """Removes axis labels and ticks for a cleaner look."""
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(axis='both', which='both', length=0)

    def add_tracked_object(self, name: str, color_condition: Callable[[np.ndarray], np.ndarray], highlight_color: str):
        """
        Registers a new object to be tracked and highlighted on the plot.
        
        Args:
            name (str): A unique name for the object (e.g., 'hand', 'target').
            color_condition (Callable): A function that returns a boolean mask for the object's pixels.
            highlight_color (str): The color for the highlight border (e.g., 'g', 'r').
        """
        if name in self.tracked_objects:
            print(f"Warning: Tracked object with name '{name}' already exists. It will be overwritten.")
        self.tracked_objects[name] = TrackedObject(name, color_condition, highlight_color)

    def _get_cell_rect_params(self, center_x: float, center_y: float) -> Tuple[float, float]:
        """
        Converts pixel coordinates to grid cell indices and calculates the drawing rectangle's corner.
        
        For example, if center_x is 18.5 and cell_size is 4, 18.5 // 4 results in 4.
        This gives us the 0-based index of the grid cell.
        
        Args:
            center_x (float): The x-coordinate of the object's center in pixels.
            center_y (float): The y-coordinate of the object's center in pixels.
            
        Returns:
            Tuple[float, float]: The (x, y) coordinates for the bottom-left corner of the rectangle patch.
        """
        cell_x = int(center_x // self.cell_size)
        cell_y = int(center_y // self.cell_size)
        rect_x = cell_x * self.cell_size - 0.5
        rect_y = cell_y * self.cell_size - 0.5
        return rect_x, rect_y

    def update(self, frame: np.ndarray):
        """
        Updates the plot with a new frame and highlights all registered tracked objects.
        
        Args:
            frame (np.ndarray): The new image frame to display.
        """
        self.img_display.set_data(frame)
        
        for obj in self.tracked_objects.values():
            # Remove the previous highlight for this object, if it exists
            if obj.patch:
                obj.patch.remove()
                obj.patch = None
            
            # Find the object in the current frame
            center = find_object_center(frame, obj.color_condition)
            
            # If found, draw a new highlight
            if center:
                center_x, center_y = center
                rect_x, rect_y = self._get_cell_rect_params(center_x, center_y)
                
                new_patch = patches.Rectangle(
                    (rect_x, rect_y), self.cell_size, self.cell_size,
                    linewidth=2, edgecolor=obj.highlight_color, facecolor='none'
                )
                self.ax.add_patch(new_patch)
                obj.patch = new_patch
                
        # Redraw the canvas to show all updates
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Closes the interactive plot window."""
        plt.ioff()
        plt.show()

# --- Example Usage ---
# This part demonstrates how the refactored class would be used.
if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_SIZE = 28
    CELL_SIZE = 4
    
    # --- Create a Plotter ---
    plotter = InteractivePlot(image_size=IMAGE_SIZE, cell_size=CELL_SIZE)
    
    # --- Define and Register Objects to Track ---
    hand_condition = lambda frame: (frame[:, :, 2] > frame[:, :, 0]) & (frame[:, :, 2] > frame[:, :, 1]) & (frame[:, :, 2] > 0.1)
    target_condition = lambda frame: (frame[:, :, 0] > frame[:, :, 1]) & (frame[:, :, 0] > frame[:, :, 2]) & (frame[:, :, 0] > 0.1)
    
    plotter.add_tracked_object(name='hand', color_condition=hand_condition, highlight_color='g')
    plotter.add_tracked_object(name='target', color_condition=target_condition, highlight_color='r')

    # --- Simulation Loop ---
    try:
        # Create a dummy environment frame
        frame = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 0.2
        
        # Place a 'hand' (blue) and 'target' (red)
        frame[10:12, 15:17, 2] = 0.9 # Blue square
        frame[20:22, 5:7, 0] = 0.9  # Red square
        
        print("Displaying initial frame. Press Ctrl+C to exit.")
        plotter.update(frame)
        plt.pause(2)
        
        # Simulate movement
        print("Simulating object movement...")
        for i in range(5):
            # Move hand
            frame[10+i:12+i, 15-i:17-i, 2] = 0.0
            frame[10+i+1:12+i+1, 15-i-1:17-i-1, 2] = 0.9
            
            plotter.update(frame)
            plt.pause(0.5)

    except KeyboardInterrupt:
        print("\nClosing plot.")
    finally:
        plotter.close()