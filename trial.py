import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import concurrent.futures

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

    if full_image is None:
        raise ValueError("Failed to load full image!")
    if cropped_image is None:
        raise ValueError("Failed to load cropped image!")

    return full_image, cropped_image

def enhance_contrast(image):
    if image is None:
        raise ValueError("Image is None. Cannot enhance contrast.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def extract_features(image):
    if image is None:
        raise ValueError("Image is None. Cannot extract features.")

    image = enhance_contrast(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def fitness_function(full_image, cropped_image, x, y):
    ch, cw, _ = cropped_image.shape
    h, w, _ = full_image.shape

    if x + cw > w or y + ch > h:
        return float('inf')

    part = full_image[int(y):int(y) + ch, int(x):int(x) + cw]
    kp1, desc1 = extract_features(cropped_image)
    kp2, desc2 = extract_features(part)

    if desc1 is None or desc2 is None:
        return float('inf')
    if len(kp1) == 0 or len(kp2) == 0:
        return float('inf')

    matches = match_features(desc1, desc2)
    if len(matches) == 0:
        return float('inf')

    distance = sum(m.distance for m in matches) / len(matches)
    return distance

def calculate_num_particles(full_image):
    h, w, _ = full_image.shape
    num_particles = (h*w) // 1000
    #num_particles = 100 
    return num_particles

def pso_search(full_image, cropped_image, max_iter=100, inertia=0.9, cognitive=1.5, social=1.5):
    h, w, _ = full_image.shape
    ch, cw, _ = cropped_image.shape

    num_particles = calculate_num_particles(full_image)
    particles = np.random.randint(0, [w - cw, h - ch], (num_particles, 2)).astype(float)
    velocities = np.random.uniform(-1, 1, (num_particles, 2))

    personal_best_positions = particles.copy()
    
    # Compute initial fitness scores in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fitness_function, full_image, cropped_image, int(p[0]), int(p[1])) for p in particles]
        personal_best_scores = np.array([f.result() for f in futures])

    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    inertia_decay = 0.99

    for iteration in range(max_iter):
        # Update positions and velocities
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fitness_function, full_image, cropped_image, int(p[0]), int(p[1])) for p in particles]
            scores = np.array([f.result() for f in futures])

        for i, particle in enumerate(particles):
            velocities[i] = (inertia * velocities[i] + 
                             cognitive * np.random.random() * (personal_best_positions[i] - particle) + 
                             social * np.random.random() * (global_best_position - particle))
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], [0, 0], [w - cw, h - ch])

            if scores[i] < personal_best_scores[i]:
                personal_best_scores[i] = scores[i]
                personal_best_positions[i] = particles[i]

            if scores[i] < global_best_score:
                global_best_score = scores[i]
                global_best_position = particles[i]

        print(f"Iteration {iteration + 1}/{max_iter}, Best Score: {global_best_score}")

        inertia *= inertia_decay

        if iteration % 2 == 0:
            highlight_found_position(full_image, cropped_image, global_best_position, wait_time=1)

    return global_best_position

def highlight_found_position(full_image, cropped_image, best_position, wait_time=1):
    x, y = map(int, best_position)
    ch, cw, _ = cropped_image.shape

    gray_full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    gray_full_image = cv2.cvtColor(gray_full_image, cv2.COLOR_GRAY2BGR)

    highlighted_image = gray_full_image.copy()
    highlighted_image[y:y+ch, x:x+cw] = full_image[y:y+ch, x:x+cw]

    cv2.imshow("Highlighted Image", highlighted_image)
    cv2.waitKey(wait_time)

# Main code
try:
    full_image, cropped_image = load_image_from_dialog()
    best_position = pso_search(full_image, cropped_image)
    print(f"Best position found: {best_position}")
    highlight_found_position(full_image, cropped_image, best_position, wait_time=0)
except Exception as e:
    print(f"Error: {e}")

cv2.waitKey(0)
cv2.destroyAllWindows()
