import sys
import pygame
import gymnasium as gym

sys.path.append('../../../robot/')

# Initialize pygame for keyboard input
pygame.init()

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
while running:
    env.render()  # Render environment
    
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
                print(env.unwrapped.get_screen().shape)
                
                print(f"Action Taken: {action}, Reward: {reward:.4f}")

                if terminated or truncated:
                    obs, info = env.reset(options={'only_target': True})  # Reset environment if episode ends

# Cleanup
env.close()
pygame.quit()