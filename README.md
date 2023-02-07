

#  Bachelor Thesis: Using a deep learning MLP to predict gaze Hits on sponsor messages in football stadiums 

## Work in Progress

This is an implementation of a Multi Layer Perceptron (MLP) using Python 3, Keras and TensorFlow.
The model is trained to solve a binary classification problem and uses SMOTE and an Undersampler technique 
to combat imbalanced data. A Hyperparameter Search was conducted multiple times to find parameters 
for the best accuracy.    

Its goal is to predict consumers' gaze hits on sponsoring messages in football stadiums depending on 39
game related variables.

Right now, the best accuracy was recorded at 89.85%. The Model shall be optimized, until an accuracy of at least 
96% will be recorded.

## Dataset
The Dataset is based on values conducted by [Herold et al.](https://doi.org/10.3390/su13042312), using a quantitative research design.
Broadcasts of the german Bundesliga from season 2019-2020 were presented to 26 highly involved participants. 
They were invited to watch a home game of their favorite team to ensure a sufficiently strong emotional reaction
[(2021, 7)](https://doi.org/10.3390/su13042312). 
Heart rate, eye-tracking, and betting odds data served as measurements of arousal, attention, and
game outcome uncertainty and were aggregated on a second-by-second basis (k = 140,400)[(8)](https://doi.org/10.3390/su13042312).

After clearing out missing and incorrect data a total of k = 115457 pairs remained. This was done using this [Script](/clean_data.py).

The used Dataset including all game-related factors such as Ball Possession and other Run-of-Play variables is not
included in this repository, as the describing paper is still under review.

## Model
(Coming Soon)

The repository includes: (Work in Progress)
* Source code
* logs

## Requirements
Python 3.10, TensorFlow 2.11, Keras and other common packages listed in `requirements.txt`.

## Installation
1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ```

## Logging to TensorBoard
TensorBoard is another great debugging and visualization tool. The model is configured to log losses and save weights at the end of every epoch.
(Work in Progress)

## Bibliography 
Herold, Elisa, Felix Boronczyk, und Christoph Breuer. 2021. 
„Professional Clubs as Platforms in Multi-Sided Markets in Times of COVID-19: 
The Role of Spectators and Atmosphere in Live Football“. Sustainability 13 (4): 2312. 
https://doi.org/10.3390/su13042312.
