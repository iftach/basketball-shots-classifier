Project Background:
In this project we examined whether a non-structured classifier yields a better accuarcy then a structured classifier (HMM) in
the task of classifying basketball shots.
The project is divided into 3 parts.

Basic part: we compare the accuracy of a naïve classifier, unstructured classifiers and structured classifier based on
Viterbi algorithm that we implemented.

Advance part: we researched the effect of different sequences on the accuracy of our structured classifier and whether
a more natural sequence to the problem yields better results. Usually for every problem there is a sequence that feels natural
to the problem, in speech recognition it can be a sentence in an article, in our problem it the can be the shots of a player
in a game.

Creative part: we try to look at the problem from a different angle, while accuracy is good for data scientists, what is
important is what makes a shot a good one, therefor a recall is a more suitable parameter than accuracy because if we can tell
what shots will go in, then we know what works for a player.

Prerequisites: 
A. Python3 should be installed 
B. Pip should be installed.
C. Run pip install for scipy, pandas, numpy and sklearn (pip install scipy pandas sklearn numpy) 

Running all parts:
python main.py
NOTE: main.py runs all the parts of the projet simultaneously (multiproccessing)
