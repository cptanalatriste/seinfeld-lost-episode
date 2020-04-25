# seinfeld-lost-episode

![Tom's Restautant](https://github.com/cptanalatriste/seinfeld-lost-episode/blob/master/img/restaurant.jpg?raw=true)


A Recurrent Neural Network (RNN) for generating [Seinfeld TV scripts](https://www.youtube.com/watch?v=DAT7KzyQd34).
After training on the scripts from 9 seasons, it generates the following: 

> Jerry: I know what, you want a little one of those people in the time.
>
> Elaine: Oh, yeah.....
>
> George:(to elaine) Oh, hi. i don't want to go to the hospital.(elaine enters)
>
> Elaine: Hey!
>
> George:(on phone) Hey, i got to get out of here!
>
> Kramer: Yeah, yeah, yeah, i don't know.(jerry shakes a hand to the couch)


## Getting started
To train the network, be sure to first clone this repository and install
all the Python packages defined in `requirements.txt`.

The pre-trained network is available at `trained_rnn.pt`.

## Instructions
To explore the training process, you can take a look at the `dlnd_tv_script_generation.ipynb` jupyter notebook.
The network code is contained in the `lost_episode` module.