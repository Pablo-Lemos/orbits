# orbits

Code to reproduce results from the paper "Rediscovering orbital mechanics with machine learning". 

- The folder "simulate" contains the base clases, as well as code to generate simulated orbits, and to read the nasa data 
- The folder "data" contains the data downloaded from the NASA Ephymeris system, and preprocessed into the format specified in the paper
- The file "ml_model.py" contains the graph neural network model 
- The file "planets_tf2.py" runs the graph neural network. 
- The file symbolic_regression.py" runs the symbolic regression algorithm. 
- All the plotting scripts used in the paper are contained in the "plotting" folder.

To run the code, the following packages are required: 

- TensorFlow 2: https://www.google.com/search?client=safari&rls=en&q=tensorflow&ie=UTF-8&oe=UTF-8
- graph_nets: https://github.com/deepmind/graph_nets
- PySR: https://github.com/MilesCranmer/PySR

![Orbits Demo](data/animation.gif)
