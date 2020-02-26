# Hidden-Markov-Models
Implementation of Hidden Markov model

To implement the model we have written class HMM which denotes the Hidden Markov
Model. For this class transition and emission probabilities are used as the parameters which
are specified in the files like models/two english.trans and models/two english.emit. In .emit
file, each row is having symbol which denotes the start state.
Function HMM.load is used to load the parameters from the model files in which some of
the transitions may be omitted. Such parameters maintains value 0. If conditional probabilities are not specified in model files then constructor will initialize them randomly. Inverse
method HMM.dump takes basename of output file model’s parameter are written to the
.trans and .emit files.

Required Tools

• Numpy

• Python2

• collections

• codecs

• Random

To execute code in /code directory run following code :
```
python2 hmm.py
```
Execution is accoarding to instructions specified.
Plots generated are stored in figures folder
