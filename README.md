# RLBot

<p align="center">
  <img width="512" height="158" src="https://github.com/drssoccer55/RLBot/blob/master/images/RLBot.png">
</p>

### Short Description
Saltie is a bot that uses Neural Networks and Machine Learning to learn how to play the game.
It also has tools for training bots and collecting the replays from a lot of distributed computers

### Requirements
Windows, Rocket League, Python 3, Tensorflow.



setup for our project
Must have Windows** and an NvIDIA GPU*
https://www.tensorflow.org/install/install_windows
(when asked for nvidia specifics look at the other pin if you have version < 1.5)
And the setup for the general rocket league bot is on the general discord
https://github.com/drssoccer55/RLBot/wiki/Setup-Instructions-%28current%29

*Note: if you only want to collect data or run the bot you can run it on CPU version instead.

After installing tensorflow (which we recommend via pip3)
You must install a library called requests

`pip3 install requests`

We use a library called tqdm for printing out timing info

`pip3 install tqdm`


If you only want to help train and can not generate data you must have an Nvidia GPU and Ubuntu/linux

To upload data to our server you must grab a config file from the discord

For streamers of the bot to get the fancy graphs you must install pyqtgraph

`pip3 install pyqtgraph`

`pip3 install PyQt5`

An alternative option is to install quicktracer

`pip3 install quicktracer`

To install the needed libraries to run the human_bot

`pip install inputs`
