import pygame
import time
import math
import csv
import random

# ======================
# CONFIG
# ======================
NUM_TRIALS = 15
TARGET_RADIUS = 20          # W = 40 px (diameter)
BG_COLOR = (25, 25, 25)
TARGET_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

# ======================
# INIT
# ======================
pygame.init()

# Get actual screen resolution (fixes black screen issue)
info = pygame.display.Info()
SCREEN_W, SCREEN_H = info.current_w, info.current_h

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Fitts' Law Test – Click the Green Dot")

font = pygame.font.SysFont("Arial", 26)
clock = pygame.time.Clock()

# ======================
# LOGGING
# ======================
log = open("fitts_log.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["trial", "D", "W", "MT", "hit"])

# ======================
# TARGET GENERATOR
# ======================
def random_target():
    return (
        random.randint(TARGET_RADIUS, SCREEN_W - TARGET_RADIUS),
        random.randint(TARGET_RADIUS, SCREEN_H - TARGET_RADIUS)
    )

targets = [random_target() for _ in range(NUM_TRIALS + 1)]
prev_target = targets[0]

# ======================
# STATE
# ======================
trial = 0
trial_start_time = time.time()
running = True

# ======================
# MAIN LOOP
# ======================
while running and trial < NUM_TRIALS:
    screen.fill(BG_COLOR)

    target = targets[trial + 1]
    pygame.draw.circle(screen, TARGET_COLOR, target, TARGET_RADIUS)

    label = font.render(
        f"Trial {trial + 1} / {NUM_TRIALS}",
        True,
        TEXT_COLOR
    )
    screen.blit(label, (20, 20))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            dist = math.dist((mx, my), target)

            hit = dist <= TARGET_RADIUS
            MT = time.time() - trial_start_time
            D = math.dist(prev_target, target)
            W = TARGET_RADIUS * 2

            writer.writerow([trial, D, W, MT, int(hit)])

            prev_target = target
            trial += 1
            trial_start_time = time.time()

    clock.tick(60)

# ======================
# CLEANUP
# ======================
log.close()
pygame.quit()

print("Fitts test completed.")
print("Results saved to fitts_log.csv")
