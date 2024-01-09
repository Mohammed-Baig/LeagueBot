#Voice-Controlled League of Legends Bot

##Overview
Welcome to the Voice-Controlled League of Legends Bot repository! This project enables you to control certain League of Legends characters using voice commands. The bot utilizes speech-to-text in Python, leverages the ChatGPT natural language processing (NLP) function calls to interpret commands, and integrates with YOLOv4 Tiny for live object detection to detect players in the game.

##Features
- Voice Commands: Control League of Legends characters through simple voice commands.
- NLP Integration: Utilize ChatGPT's NLP to parse the meaning of voice inputs and determine appropriate actions and parameters for in-game commands.
- Map Navigation: Move the bot to different locations on the map using objective coordinates as landmarks (scaling based on screen resolution).
- Manual Input: Users can run diagnostics and manually input coordinates for further customization.
- Object Detection: Player detection in the game is powered by YOLOv4 Tiny live object detection model.
- Easy Setup: The requirements.txt file ensures easy installation of all necessary libraries for smooth execution.

##Getting Started
1. Clone the repository
git clone https://github.com/your-username/league-of-legends-bot.git
cd league-of-legends-bot

2. Install dependencies:
pip install -r requirements.txt

3. Run the bot:
python main.py

##Contribution
Feel free to contribute to the project by adding new examples to the notebooks provided in the repository. This helps in refining the training data and improving the bot's performance.

##Credits
YOLOv4 Tiny live object detection model code: https://github.com/moises-dias/yolo-opencv-detector
Tutorials: https://www.youtube.com/watch?v=RSXgyDf2ALo&t=176s and https://youtu.be/gdIVHdRbhOs?si=4Yrk0u7o1JnWEfON

