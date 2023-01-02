World of Warcraft Location Guesser.

This program is based on the idea of taking a picture of a landscape and it guessing where you are in the world.  
A game with a large expasnive enviorment with varing different zones to play in is the game world of warcraft. 
This project aims to set up a Neural Network that takes input of folders, and interate over them to create an
model that can be used in the backend of a website.  This project examples are based off of several hundred photos
from the World of Warcraft Expansion Shadowlands.  Once the Model is run, it will can then be run the provided website
where a user can upload an image, and the website will run the picture throught the model and attempt to guess where
the location in game this was taken


To run the program
python classify.py --model wowzone.model --labelbin lb.pickle	--image examples/Bastion.png

to run website, make sure to either port forward or use ngrok to make website servicable
