import pyautogui
import time
import speech_recognition as sr
import pyttsx3
import openai
import cv2 as cv
import json
import numpy as np
import win32gui, win32ui, win32con
from PIL import Image
import os

class WindowCapture:
    w = 0
    h = 0
    hwnd = None

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[..., :3]
        img = np.ascontiguousarray(img)

        return img

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while (True):
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpeg")
            time.sleep(1)

    def get_window_size(self):
        return (self.w, self.h)

class ImageProcessor:
    W = 0
    H = 0
    net = None
    ln = None
    classes = {}
    colors = []

    def __init__(self, img_size, cfg_file, weights_file):
        np.random.seed(42)
        self.net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.W = img_size[0]
        self.H = img_size[1]

        with open('yolov4-tiny/obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()

        # If you plan to utilize more than six classes, please include additional colors in this list.
        self.colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]

    def proccess_image(self, img):

        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)

        coordinates = self.get_coordinates(outputs, 0.5)

        self.draw_identified_objects(img, coordinates)

        return coordinates

    def get_coordinates(self, outputs, conf):

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]

            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                p0 = int(x - w // 2), int(y - h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)

        if len(indices) == 0:
            return []

        coordinates = []
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            coordinates.append(
                {'x': x, 'y': y, 'w': w, 'h': h, 'class': classIDs[i], 'class_name': self.classes[classIDs[i]]})
        return coordinates

    def draw_identified_objects(self, img, coordinates):
        for coordinate in coordinates:
            x = coordinate['x']
            y = coordinate['y']
            w = coordinate['w']
            h = coordinate['h']
            classID = coordinate['class']

            color = self.colors[classID]

            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, self.classes[classID], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow('DETECTED OBJECTS', img)

#Parse data in lines to extract proper key:value pairs and ensure list data integrity remains for values for keys
def parse_line(line):
    key, value = line.strip().split(":")
    key = key.strip().strip('"')
    value = [float(coord) for coord in value.strip().strip('[],').split(',')]
    return key, value

#Reads lines from text file to convert to a dictionary
def read_dictionary_from_file(file_path):
    result_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            key, value = parse_line(line)
            result_dict[key] = value
    return result_dict

#Text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

#Diagnostic to tell the user about everything they're going to need, and customize coordinates if needed
def diagnostic():
    #Logged locations of each major landmark on summoners rift
    example_dictionary = {
        "blue nexus": [1, 2],
        "blue top nexus turret": [2, 3],
        "blue bottom nexus turret": [3, 4],
        "blue top inhibitor": [4, 6],
        "blue middle inhibitor": [5, 7],
        "blue bottom inhibitor": [4, 6],
        "blue top inhibitor turret": [4, 6],
        "blue middle inhibitor turret": [4, 6],
        "blue bottom inhibitor turret": [4, 6],
        "blue top inner turret": [4, 6],
        "blue middle inner turret": [4, 6],
        "blue bottom inner turret": [4, 6],
        "blue top outer turret": [4, 6],
        "blue middle outer turret": [4, 6],
        "blue bottom outer turret": [4, 6],
        "red nexus": [1, 2],
        "red top nexus turret": [2, 3],
        "red bottom nexus turret": [3, 4],
        "red top inhibitor": [4, 6],
        "red middle inhibitor": [5, 7],
        "red bottom inhibitor": [4, 6],
        "red top inhibitor turret": [4, 6],
        "red middle inhibitor turret": [4, 6],
        "red bottom inhibitor turret": [4, 6],
        "red top inner turret": [4, 6],
        "red middle inner turret": [4, 6],
        "red bottom inner turret": [4, 6],
        "red top outer turret": [4, 6],
        "red middle outer turret": [4, 6],
        "red bottom outer turret": [4, 6],
        "baron": [4, 6],
        "baron scuttle": [4, 6],
        "baron blue": [4, 6],
        "baron gromp": [4, 6],
        "baron wolves": [4, 6],
        "baron red": [4, 6],
        "baron raptors": [4, 6],
        "baron krugs": [4, 6],
        "dragon": [4, 6],
        "dragon scuttle": [4, 6],
        "dragon blue": [4, 6],
        "dragon gromp": [4, 6],
        "dragon wolves": [4, 6],
        "dragon red": [4, 6],
        "dragon raptors": [4, 6],
        "dragon krugs": [4, 6],
        "item purchase bar": [4, 6],
        "item location": [4, 6]
    }

    #Pre installation requirements
    print(
        "Pre installation requirements:\n1) Download opencv, pyautogui\n2) Resize in game map(map scale to 100)\n"
        "3) Turn Quick Cast on\n4) Attack move(https://youtu.be/-oyxOgtT33U?si=htv6opgAgOARQfkz and bind to A)\n"
        "5) Lock the camera on character\n6) Get coordinates of in game map positions\n"
        "7) Get coordinates of search bar position item location(slightly under)\n"
    )

    #Confirmation
    proceed = int(input("If you would like to proceed with the diagnostic press 1, else press 2:"))
    if (proceed == 1):
        print(
            "Steps 1-5 must be done externally. For the coordinates of each landmark it is recommended that you have 2"
            " in game screenshots instead of attempting to run the coordinate checker live. The screenshots needed"
            " are 2 regular ones, one of the normal in game screen, and the other with the shop open, and the "
            " search bar and item that pops up clearly visible. To do so open up the game and press F12 or FN + F12 "
            " depending on your machine, then navigate to the screenshots folder"
            " in your league of legends folder, open the screenshots in Photos, click on the 1:1 button and then click"
            " full screen. From here follow the instructions below. The manual calibration will take approximately"
            " 10 minutes. \n")

        #Go through each entry in the example_dictionary dictionary to custom enter each field
        for objective in example_dictionary:
            proceed = int(input("To get the coordinates for " + objective + " press 1, then navigate to your screenshot"
                                                                           " and hover your mouse over its location on"
                                                                           " the map and wait 5 seconds. Please press"
                                                                           " 1 whenever you're ready: "))
            if proceed == 1:
                time.sleep(5)
                x,y = pyautogui.position()
            example_dictionary[objective] = [x,y]
            print(f"Coordinates for {objective}: {example_dictionary[objective]}\n")

        #Write custom entered coordinates to text file for later use
        file_path = "custom_specialized_coordinates.txt"
        with open(file_path, 'w+') as file:
            for key, value in example_dictionary.items():
                line = f'"{key}": {value},\n'
                file.write(line)

    else:
        quit()

#Confirm with the user to see if they would like to continue or
def confirm_return():
    y = int(input("would you like to\n1.return to main menu\n2.quit\n"))
    if (y == 1):
        main()

    elif (y == 2):
        quit()

    else:
        print("invalid input, please try again later")
        quit()

#Moves to specified specified champion and follow them
def move_to_champion(coordinates, champion_name):
    champion_coordinates = [c for c in coordinates if c["class_name"] == champion_name]
    champion_to_move_to = champion_coordinates[0]
    pyautogui.moveTo(champion_to_move_to['x'], champion_to_move_to['y'])
    pyautogui.click(champion_to_move_to['x'], champion_to_move_to['y'])

#Moves to specified location in example_dictionary based on map landmarks
def move_to_landmark(location_dict, location_name):
    if location_name in location_dict:
        coordinates = location_dict[location_name]
    x,y = coordinates[0], coordinates[1]
    pyautogui.moveTo(x, y)
    pyautogui.rightClick(x, y)

#Uses an ability, summoner spells, or items. Just specify button it's assigned to
def use_ability(ability):
    pyautogui.press(ability)

#Purchases the item
def purchase_item(location_dict, item_name):
    pyautogui.press('p')
    if "item purchase bar" in location_dict:
        x, y = location_dict["item purchase bar"]
        pyautogui.moveTo(x, y)
        pyautogui.leftClick(x, y)
        pyautogui.write(item_name)
    if "item location" in location_dict:
        x, y = location_dict["item location"]
        pyautogui.moveTo(x, y)
        pyautogui.doubleClick(x, y)

#level ability
def level_up(input):
    pyautogui.hotkey('alt', input)

#Attacks the nearest enemy based on keybindings
def auto_attack():
    pyautogui.press('a')

#Recall on command
def recall():
    pyautogui.press('b')

functions = [
    {
        'name': 'move_to_champion',
        'description': 'Moves to champion given champion name(champion_name) and x,y list of coordinates(coordinates)',
        'parameters':
        {
            'type': 'object',
            'properties': {
                'coordinates': {
                    'type:': 'dictionary',
                    'description': 'Key: value pair where the key is the champion name and the values are x,y coordinates. Not part of the user input but declared before in the main method of the code and called in this function'
                },
                'champion_name': {
                    'type': 'string',
                    'description': 'Name of the champion the user is going to be looking for'
                }
            },
            'required': ['coordinates', 'champion_name']
        }
    },
    {
        'name': 'move_to_landmark',
        'description': 'Moves to a landmark on the map given landmark name(location_name) and its x,y coordinates in (location_dict)',
        'parameters':
        {
            'type': 'object',
            'properties': {
                'location_dict': {
                    'type': 'dictionary',
                    'description': 'key:value pairs where the location_name is the key and the values are x,y coordinatess in [x,y] format. Not part of the user input but declared before in the method of the code and called in this function'
                },
                'location_name': {
                    'type:': 'string',
                    'description': 'name of the location, used as the key in location_dict to get x,y values'
                }
            },
            'required': ['location_name', 'location_dict']
        }
    },
    {
        'name': 'use_ability',
        'description': 'uses an ability, or item. Abilities are represented by the letters[Q,W,E,R] and items are numbers[1,2,3,4,5,6]',
        'parameters':
        {
            'type': 'object',
            'properties': {
                'ability': {
                    'type:': 'string',
                    'description': 'the letter or number corresponding to the ability. The user says it in the command and that is what is pressed'
                }
            },
            'required': ['ability']
        }
    },
    {
        'name': 'purchase_item',
        'description': 'opens the shops and purchases the item specified by the user. Searches for the item name(item_name) in the search box, and gets the coordinates of the search box and item from the location_dict dictionary. This again not part of the user input but declared and passed in before the function call in the main method',
        'parameters':
        {
            'type': 'object',
            'properties': {
                'location_dict': {
                    'type:': 'dictionary',
                    'description': 'key:value pairs where the location of the search box and the item are keys with their own corresponding [x,y] coordinate value pairss. Not part of the user input but declared before in the method of the code and called in this function'
                },
                'item_name': {
                    'type:': 'string',
                    'description': 'name of the item that the user wants to purchase'
                }
            },
            'required': ['item_name', 'location_dict']
        }
    },
    {
        'name': 'level_up',
        'description': 'takes the input[Q,W,E,R] and pressed it and ALT to level up the corresponding ability',
        'parameters':
        {
            'type': 'object',
            'properties': {
                'input': {
                    'type:': 'string',
                    'description': 'a string of the letter representative of the ability the user wants to level further'
                }
            },
            'required': ['input']
        }
    },
    {
        'name': 'auto_attack',
        'description': 'presses the A key to auto attack the nearest enemy based off of keybinds',
        'parameters':
        {
            'type': 'object',
            'properties': {
            },
            'required': ['']
        }
    },
    {
        'name': 'recall',
        'description': 'presses the B key to initiate a recall and return to base/home',
        'parameters':
        {
            'type': 'object',
            'properties': {
            },
            'required': ['']
        }
    }
]

available_functions = {
    'move_to_champion': move_to_champion,
    'move_to_landmark': move_to_landmark,
    'use_ability': use_ability,
    'purchase_item': purchase_item,
    'level_up': level_up,
    'auto_attack': auto_attack,
    'recall': recall
}

def main():

    #Initialize openAI API Key so it can read functions from speech commands
    openai.api_key = open('API_KEY', 'r').read()

    #Initial Speech Recognition Recognizer
    r = sr.Recognizer()

    #Read normalized coordinates
    normalized_coords = read_dictionary_from_file("preset_normalized_coordinates.txt")

    #Get screen resolution
    x, y = pyautogui.size()

    #Scale coordinates given screen resolution
    for location in normalized_coords:
        x1,y1 = normalized_coords[location]
        x_final = int(x1 * x)
        y_final = int(y1 * y)
        normalized_coords[location] = [x_final, y_final]

    print("This program allows you to play league of legends using entirely voice commands. Note your game must be set"
          " to Windowed mode in order for the program to work")

    x = int(input(
        "1) play game\n2) run diagnostic\n3) view available champions\n4) view preset settings\nSelect here: "))

    if (x == 1):
        conf = int(input("Before we proceed we must confirm whether you would like to use the preset coordinates or your "
                  " own custom ones. Press 1 if you would like to use presets, Press 2 if you would like to use"
                  " your custom set ones: "))

        if(conf == 1):
            coordinate_dictionary = normalized_coords

        elif(conf == 2):
            coordinate_dictionary = read_dictionary_from_file("custom_specialized_coordinates.txt")

        else:
            print("Invalid input. Please try again.")
            quit()

        window_name = "[INSERT WINDOWED LEAGUE CLIENT NAME HERE]"
        cfg_file_name = "./yolov4-tiny/yolov4-tiny-custom.cfg"
        weights_file_name = "yolov4-tiny-custom_last.weights"

        wincap = WindowCapture(window_name)
        improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)
        while True:
            ss = wincap.get_screenshot()
            coordinates = improc.proccess_image(ss)
            try:
                #device_index set to 2 because my microphone doesn't work and so I had to use my headphone mic
                with sr.Microphone(device_index=2) as source2:
                    r.adjust_for_ambient_noise(source2, duration=0.2)
                    audio2 = r.listen(source2)
                    MyText = r.recognize_google(audio2)
                    MyText = MyText.lower()

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": MyText},
                    ],
                    functions=functions,
                    function_call='auto'
                )

                response_message = response["choices"][0]["text"]

                if response_message.get('function_call'):
                    function_name = response_message['function_call']['name']
                    function_args = json.loads(response_message['function_call']['arguments'])

                    if function_name == "move_to_champion":
                        args_dict = {'champion_name': function_args.get('champion_name')}
                        move_to_champion(coordinates, **args_dict)

                    elif function_name == "move_to_landmark":
                        args_dict = {'location_name': function_args.get('location_name')}
                        move_to_landmark(coordinate_dictionary, **args_dict)

                    elif function_name == "use_ability":
                        args_dict = {'ability': function_args.get('ability')}
                        use_ability(**args_dict)

                    elif function_name == "purchase_item":
                        args_dict = {'item_name': function_args.get('item_name')}
                        purchase_item(coordinates, **args_dict)

                    elif function_name == "level_up":
                        args_dict = {'input': function_args.get('input')}
                        level_up(**args_dict)

                    elif function_name == "auto_attack":
                        auto_attack()

                    elif function_name == "recall":
                        recall()

            except sr.UnknownValueError:
                print("unknown error occurred")

    #Run the diagnostic
    elif (x == 2):
        diagnostic()
        confirm_return()

    #Show available champions
    elif (x == 3):
        print("Available champions\n"
              "top: jax, nasus, trynda, garen, darius\n"
              "support: yuumi, sona\n"
              "adc: sivir\n")

        confirm_return()

    #Show the current coordinates for the users resolution
    elif (x == 4):
        print("Coordinates are normalized for all screen resolutions, there are the settings for your screen:")
        print(normalized_coords)
        confirm_return()

    else:
        print("Invalid input, please try again later")
        quit()

if __name__ == "__main__":
    main()