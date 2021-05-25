* Moved the network definition to another method, which makes subclassing easier
* Parametrized everything (batch_size, initial_epsilon,...)
* Parametrized epsilon decay type: smooth or flat (default to flat). Smooth one normalizing operation is done automatically: no need to specify special epsilon_decay value, we can just specify 1e-3 and the code will auto compute 1-epsilon_decay when using smooth decay
* Changed save and load so that also hyperparameters are loaded and keras functions are used (they also save optimizer things.. pretty useful to continue learning later)
* Moved feature processing (through the dense layers) in another method so that both advantage and feed forward methods are decoupled from it (they do the same thing)
* decoupled agent class from algorithm clas