import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

def load_image_from_dialog():
    root = tk.Tk()
    root.withdraw()

    full_image_path = filedialog.askopenfilename(title="Select the Full Image")
    if not full_image_path:
        raise ValueError("No full image selected!")

    cropped_image_path = filedialog.askopenfilename(title="Select the Cropped Image")
    if not cropped_image_path:
        raise ValueError("No cropped image selected!")

    full_image = cv2.imread(full_image_path)
    cropped_image = cv2.imread(cropped_image_path)

    if full_image is None or cropped_image is None:
        raise ValueError("Error loading images!")

    return full_image, cropped_image

def fitness_function(full_image, cropped_image, x, y):
    ch, cw, _ = cropped_image.shape
    h, w, _ = full_image.shape

    # Ensure the bounding box is within the image boundaries
    if x + cw > w or y + ch > h:
        return float('inf')  # Return a high error if the bounding box is out of bounds

    part = full_image[int(y):int(y) + ch, int(x):int(x) + cw]
    mse = ((part - cropped_image) ** 2).mean()
    return mse

def calculate_num_particles(full_image):
    h, w, _ = full_image.shape
    num_particles =  ((h * w) // 1000)
    return num_particles

def pso_search(full_image, cropped_image, threshold, max_iter=100, inertia=0.9, cognitive=1.5, social=1.5, stagnation_threshold=20):
    h, w, _ = full_image.shape
    ch, cw, _ = cropped_image.shape

    num_particles = calculate_num_particles(full_image)

    particles = np.random.randint(0, [w - cw, h - ch], (num_particles, 2)).astype(float)
    velocities = np.random.uniform(-1, 1, (num_particles, 2))

    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_function(full_image, cropped_image, x, y) for x, y in particles])

    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)
    previous_global_best_score = global_best_score

    inertia_decay = 0.99
    stagnation_counter = 0

    iteration = 0
    while iteration < max_iter and global_best_score > threshold:
        for i, particle in enumerate(particles):
            velocities[i] = inertia * velocities[i] + \
                            cognitive * np.random.random() * (personal_best_positions[i] - particle) + \
                            social * np.random.random() * (global_best_position - particle)
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], [0, 0], [w - cw, h - ch])

            score = fitness_function(full_image, cropped_image, int(particles[i][0]), int(particles[i][1]))
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

            if score < global_best_score:
                global_best_score = score
                global_best_position = particles[i]
                stagnation_counter = 0
            elif score == previous_global_best_score:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

        previous_global_best_score = global_best_score

        print(f"Iteration {iteration + 1}/{max_iter}, Best Score: {global_best_score}")

        if stagnation_counter >= stagnation_threshold:
            print(f"Global best score stagnant for {stagnation_threshold} iterations. Resetting swarm positions.")
            particles = np.random.randint(0, [w - cw, h - ch], (num_particles, 2)).astype(float)
            velocities = np.random.uniform(-1, 1, (num_particles, 2))
            stagnation_counter = 0

        inertia *= inertia_decay
        iteration += 1

        if iteration % 2 == 0:
            highlight_found_position(full_image, cropped_image, global_best_position, wait_time=1)

    if global_best_score > threshold:
        print("Threshold not met after max iterations. Rerunning PSO.")
        return pso_search(full_image, cropped_image,threshold, max_iter, inertia, cognitive, social, stagnation_threshold)
    else:
        return global_best_position
    
def highlight_found_position(full_image, cropped_image, best_position, wait_time=0):
    x, y = map(int, best_position)
    ch, cw, _ = cropped_image.shape

    gray_full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    gray_full_image = cv2.cvtColor(gray_full_image, cv2.COLOR_GRAY2BGR)

    highlighted_image = gray_full_image.copy()
    highlighted_image[y:y+ch, x:x+cw] = full_image[y:y+ch, x:x+cw]

    cv2.imshow("Highlighted Image", highlighted_image)
    
    cv2.waitKey(wait_time)

def show_best_position_prompt(best_position):
    x, y = map(int, best_position)
    message = f"Best position found at (x={x}, y={y})."
    messagebox.showinfo("Best Position Found", message)

def calculate_baseline_mse(full_image, cropped_image):
    h, w, _ = full_image.shape
    ch, cw, _ = cropped_image.shape

    # Select a random region from the full image
    x = np.random.randint(0, w - cw)
    y = np.random.randint(0, h - ch)

    random_region = full_image[y:y+ch, x:x+cw]
    baseline_mse = ((random_region - cropped_image) ** 2).mean()

    return baseline_mse

def calculate_relative_scaling(full_image, cropped_image):
    full_area = full_image.shape[0] * full_image.shape[1]
    cropped_area = cropped_image.shape[0] * cropped_image.shape[1]
    return cropped_area / full_area
    





# Main code
full_image, cropped_image = load_image_from_dialog()

baseline_mse = calculate_baseline_mse(full_image, cropped_image)
relative_scaling = calculate_relative_scaling(full_image, cropped_image)
dynamic_threshold = baseline_mse/relative_scaling  # Adjust the scaling factor as needed
print(dynamic_threshold)

best_position = pso_search(full_image, cropped_image, dynamic_threshold)
print(f"Best position found: {best_position}")
highlight_found_position(full_image, cropped_image, best_position, wait_time=0)
show_best_position_prompt(best_position)


cv2.destroyAllWindows()
