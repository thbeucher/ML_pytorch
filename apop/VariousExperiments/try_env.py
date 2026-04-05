import sys
import pygame
import gymnasium as gym

from replay_buffer import ReplayBuffer
from interactive_plot import InteractivePlot

sys.path.append('../../../robot/')

# Initialize pygame for keyboard input
pygame.init()

# prepare_data
rb = ReplayBuffer(2, 1, 256, resize_to=32, normalize_img=True)

# Create environment
env = gym.make("gymnasium_env:RobotArmEnv", render_mode="human")
obs, info = env.reset()

# Define action mapping
KEY_ACTIONS = {
    pygame.K_UP: 1,      # INC_J1
    pygame.K_DOWN: 2,    # DEC_J1
    pygame.K_LEFT: 3,   # INC_J2
    pygame.K_RIGHT: 4,    # DEC_J2
}

running = True

# Initialize interactive plot
image_size = 32
cell_size = 4
plot = InteractivePlot(image_size, cell_size)

# --- Define and Register Objects to Track ---
# The color conditions are defined as lambda functions that check for specific color characteristics in the frame.
# For example, the 'hand' condition checks for pixels where the blue channel is dominant and above a certain threshold,
# while the 'target' condition checks for pixels where the red channel is dominant.
hand_condition = lambda frame: (frame[:, :, 2] > frame[:, :, 0]) & (frame[:, :, 2] > frame[:, :, 1]) & (frame[:, :, 2] > 0.1)
target_condition = lambda frame: (frame[:, :, 0] > frame[:, :, 1]) & (frame[:, :, 0] > frame[:, :, 2]) & (frame[:, :, 0] > 0.1)

plot.add_tracked_object(name='hand', color_condition=hand_condition, highlight_color='g')
plot.add_tracked_object(name='target', color_condition=target_condition, highlight_color='r')

while running:
    # Capture keyboard events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Close window
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # Quit when 'Q' is pressed
                running = False
            elif event.key in KEY_ACTIONS:
                action = KEY_ACTIONS[event.key]
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update the display with the new frame
                frame = env.render()
                _, _, frame, _, _, _, _ = rb.prepare_data(None, None, frame, None, None, None, None)
                
                # Update the plot
                plot.update(frame.squeeze().permute(1, 2, 0).numpy())
                
                print(f"Action Taken: {action}, Reward: {reward:.4f}")

                if terminated or truncated:
                    obs, info = env.reset(options={'only_target': True})  # Reset environment if episode ends

# Cleanup
plot.close()
env.close()
pygame.quit()