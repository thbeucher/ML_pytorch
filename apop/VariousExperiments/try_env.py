import sys
import pygame
import gymnasium as gym

from replay_buffer import ReplayBuffer
from interactive_plot import InteractivePlot, HAND_CONDITION, TARGET_CONDITION

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

# Initialize interactive plot for ViT patch visualization
IMAGE_SIZE = 32
PATCH_SIZE = 2  # Using 4x4 patches, so there's an 8x8 grid of patches
plot = InteractivePlot(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)

# --- Define and Register Objects to Track ---
plot.add_tracked_object(name='hand', color_condition=HAND_CONDITION, highlight_color='b')
plot.add_tracked_object(name='target', color_condition=TARGET_CONDITION, highlight_color='r')

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
                
                # Update the plot and calculate patch indices
                processed_frame = frame.squeeze().permute(1, 2, 0).numpy()
                plot.update(processed_frame)
                
                # Retrieve and print the patch indices for the ViT model
                hand_idx = plot.get_patch_index('hand')
                target_idx = plot.get_patch_index('target')
                
                print(f"Action: {action}, Reward: {reward:.4f} | Hand Index: {hand_idx}, Target Index: {target_idx}")

                if terminated or truncated:
                    obs, info = env.reset(options={'only_target': True})  # Reset environment if episode ends

# Cleanup
plot.close()
env.close()
pygame.quit()