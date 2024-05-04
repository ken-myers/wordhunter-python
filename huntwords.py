import sys
import cv2
import numpy as np
import functools      
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import words
from pygtrie import CharTrie
from copy import copy
import itertools
import time
import keyboard
import argparse

class ParseException(Exception):
    def __init__(self, message):
        super().__init__(message)

# Alternative to csv.imshow because it doesnt work on my ubuntu system for some reason
def display_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.draw()
    plt.pause(0.1)  # Pauses for 0.1 seconds to update the display
    plt.clf()  # Clears the current figure to prepare for the next image

def is_square(w, h):
    return 0.8 <= w / h <= 1.25

def word_list_nltk():
    nltk.download("words")
    return words.words()

def word_list_dict_file(filepath):
    with open(filepath) as f:
        return f.read().splitlines()

def get_squares(contours, eps=0.02):
    for contour in contours:
        epsilon = eps * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            _, _, w, h = cv2.boundingRect(approx)
            # Ensure the contour is somewhat square-shaped
            if is_square(w, h):
                yield approx

def build_trie(grid):
    #flatten grid
    flat_grid= grid.flatten()

    charset= set(flat_grid)    
    
    alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    absent_chars = alphabet - charset

    word_list =word_list_dict_file("dictionary.txt")

    #build prefix tree
    tree = CharTrie()

    for word in word_list:
        word = word.upper()
        if any(char in absent_chars for char in word):
            continue
        tree[word] = True
    

    return tree
            
def hunt_words(grid):
    
    # Recursive generator function to find words starting from a given position
    def hunt_words_from(grid, x, y, trie, prefix=None, visited=None):
        if prefix is None:
            prefix = ""
        if visited is None:
            visited = set()
        
        prefix += grid[x, y]
        visited.add((x, y))

        # check if prefix is a valid word
        if trie.has_key(prefix) and len(prefix) > 2:
            yield prefix

        #check if prefix is a valid prefix
        if not trie.has_subtrie(prefix):
            return

        # Find all possible positiions to move to
        w, h = grid.shape
        moves = [(x+dx, y+dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if 0 <= x+dx < w and 0 <= y+dy < h and (x+dx, y+dy) not in visited]

        for move in moves:
            new_x, new_y = move
            yield from hunt_words_from(grid, new_x, new_y, trie, ""+prefix, copy(visited))

    trie = build_trie(grid)
    w, h = grid.shape
    words = set()

    for x, y in itertools.product(range(w), range(h)):
        words.update(hunt_words_from(grid, x, y, trie))
    
    return words

def fuzzy_contour_pos_compare(a, b, threshold):
    # Get bounding rectangles
    ax, ay, aw, ah = cv2.boundingRect(a)
    bx, by, _, _ = cv2.boundingRect(b)

    #Compare Y first
    if abs(ay - by) > threshold * ah:
        if ay < by:
            return -1
        else:
            return 1
    # Then X
    elif abs(ax - bx) > threshold * aw:
        if ax < bx:
            return -1
        else:
            return 1
    else:
        return 0


def try_get_tile_contours(square_contours, start):

    AREA_THRESHOLD = 0.1
    VALID_SIDE_LENGTHS = [4, 5]
    PADDING_THRESHOLD = 0.1
    POSITION_THRESHOLD = 0.2

    def try_find_under(first_idx):
        #Coords of first square
        x, y, w, h = cv2.boundingRect(square_contours[first_idx])
        for i in range(first_idx+1, len(square_contours)):
            x2, y2, w2, h2 = cv2.boundingRect(square_contours[i])
            
            #Check that X is within threshold
            x_good = abs(x - x2) < w * POSITION_THRESHOLD
            
            #Check Y padding
            y_good = y2 - y > h * PADDING_THRESHOLD

            if x_good and y_good:
                return i
        return None
    
    def try_find_right(first_idx):
        #Coords of first ne
        x, y, w, h = cv2.boundingRect(square_contours[first_idx])
        for i in range(first_idx+1, len(square_contours)):
            x2, y2, w2, h2 = cv2.boundingRect(square_contours[i])
            
            #Check that Y is within threshold
            y_good = abs(y - y2) < h * POSITION_THRESHOLD
            
            #Check X padding
            x_good = x2 - x > w * PADDING_THRESHOLD

            if y_good and x_good:
                return i
        return None

    def try_build_row(length, start):
        row = [start]
        for _ in range(1, length):
            next_idx = try_find_right(row[-1])
            if next_idx is None:
                return None
            row.append(next_idx)
        return row

    start_contour = square_contours[start]
    contours = square_contours[start:]
    similar_contours = []

    #Filter for only similarly sized squares
    for contour in contours:
        a1 = cv2.contourArea(start_contour)
        a2 = cv2.contourArea(contour)
        if a1 == 0:
            continue
        if abs(a1 - a2) / a1 < AREA_THRESHOLD:
            similar_contours.append(contour)
    
    #Try to build a grid
    grid = []
    for length in reversed(VALID_SIDE_LENGTHS):
        
        candidate_grid = []

        #Try to build a top row
        row_start = start
        top_row = try_build_row(length, row_start)
        if top_row is None:
            continue
        
        candidate_grid.append(top_row)

        for i in range(1, length):
            next_row_start = try_find_under(row_start)
            if next_row_start is None:
                break
            next_row = try_build_row(length, next_row_start)
            if next_row is None:
                break
            candidate_grid.append(next_row)
            row_start = next_row_start
        
        if len(candidate_grid) == length:
            grid = candidate_grid
            break

    if len(grid) == 0:
        return None

    #Flatten
    idx_list = [contour for row in grid for contour in row]

    #Get contours
    tile_contours = [square_contours[idx] for idx in idx_list]
    return tile_contours

def to_img_list(image):

    FUZZY_POS_THRESHOLD = 0.1
    BINARY_THRESH = 100

    # Threshold Images
    _, image = cv2.threshold(image, BINARY_THRESH, 255, cv2.THRESH_BINARY)


    # Find all contours
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    square_contours = get_squares(contours, 0.02)

    # Sort squares left to right, top to bottom
    key = functools.cmp_to_key(lambda a, b: fuzzy_contour_pos_compare(a, b, FUZZY_POS_THRESHOLD))
    square_contours = sorted(square_contours, key=key)

    if(len(square_contours) == 0):
        raise ParseException("No squares found")

    for i in range(len(square_contours)):
        tile_contours = try_get_tile_contours(square_contours, i)
        if tile_contours is not None:
            break
    
    if tile_contours is None:
        raise ParseException("No board found")

    # Extract tile images
    
    tile_imgs = []
    for contour in tile_contours:
        x, y, w, h = cv2.boundingRect(contour)
        tile_imgs.append(image[y:y+h, x:x+w])
    
    return tile_imgs

def get_char_template_matching(img):

    def crop_margin(to_crop, margin):
        height, width = to_crop.shape
        crop_margin_width = int(width * margin)
        crop_margin_height = int(height * margin)

        # Crop image
        return to_crop[crop_margin_height:height-crop_margin_height, crop_margin_width:width-crop_margin_width]

    templates = {char: cv2.imread(f"templates/{char}.png", cv2.IMREAD_GRAYSCALE) for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

    best_match = None
    best_score = 0.0

    for char, template in templates.items():
        #apply same preprocessing to template and image
        
        # Crop image
        img_crop = crop_margin(img, 0.05)
        template_crop = crop_margin(template, 0.05)

        #scale to 32X32
        img_crop = cv2.resize(img_crop, (32, 32), interpolation=cv2.INTER_CUBIC)
        template_crop = cv2.resize(template_crop, (32, 32), interpolation=cv2.INTER_CUBIC)

        # Apply template matching
        res = cv2.matchTemplate(img_crop, template_crop, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)

        if score > best_score:
            best_score = score
            best_match = char
    
    return best_match

def get_char(img):
    return get_char_template_matching(img)

def get_char_tesseract(img):

    #Preprocess image

    # Find 5% margin
    height, width = img.shape
    crop_margin_width = int(width * 0.05)
    crop_margin_height = int(height * 0.05)

    # Crop image
    img = img[crop_margin_height:height-crop_margin_height, crop_margin_width:width-crop_margin_width]

    #scale up to 100
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)

    # Extract character
    CONFIG = r"--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pil_image = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_image, config=CONFIG)
    
    char = text[0]
    if char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        display_image(img)
        raise ParseException(f"Unable to find character in tile")

    return char        


def to_char_grid(img_list):
    chars = [get_char(img) for img in img_list]


    #Convert to grid
    grid_size = int(np.sqrt(len(chars)))
    char_grid = np.array(chars).reshape(grid_size, grid_size)

    return char_grid

def word_score(word):
    base_scores = [0, 0, 0, 100, 400, 800, 1400]
    if len(word) <=6:
        return base_scores[len(word)]
    else:
        return base_scores[6] + 200 * (len(word) - 6)

def do_output(words, n):
    word_scores = [(word, word_score(word)) for word in words]
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

    print(f"Top {n} words:")
    for word, score in word_scores[:min(n, len(word_scores))]:
        print(f"{word}: {score}")
    
    print(f"Found {len(words)} words for a total of {sum(score for _, score in word_scores)} points")

def wait_for_key(key, timeout=1):
    start_time = time.perf_counter()
    while True:
        if time.perf_counter() - start_time > timeout:
            return False
        if keyboard.is_pressed(key):
            return True
        time.sleep(max(timeout/10, 0.1))

def main():

    parser = argparse.ArgumentParser(description='GamePigeon Word Hunt solver.')
    parser.add_argument('-f', '--file', type=str, help='Specify the path to the screenshot of the board.')
    parser.add_argument('-n', type=int, default=10, help='Specifies the number of best-scoring words to display. Defaults to 10.')
    
    args = parser.parse_args()

    image_path = args.file
    if image_path is not None:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except FileNotFoundError:
            print("File not found")
            sys.exit(1)
        
        img_list = to_img_list(img)
    else:
        #Look for webcam
        # 
        webcam = cv2.VideoCapture(0)
        while True:
            _, frame = webcam.read()
            display_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                img_list = to_img_list(frame)
                break
            except ParseException as e:
                time.sleep(0.03)
    
    char_grid = to_char_grid(img_list)
    words = hunt_words(char_grid)
        
    do_output(words, args.n)
            
            

if __name__ == "__main__":
    main()