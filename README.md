# Devoir MultiLayer Neurale network from scratch

I created that project for my final exam at ENSET-M, Machine learning.

Video explain the code source of my project : [Code source explained]()

## What is the project about

- Solving XOR problem using MLNN
- Build MLNN from scratch 👇:
  - With parameters (Learning rates,epoches)
  - Train it using Normale distribution N(0,1)
  - Usign Mutliple activation functions and backpropagation algorithm

## Project Dependencies

Project created using :

- Conda Enviromment
- Conda vscode
- Python
- See requirement.txt

## Project Architecture :

Project Tree :

```
C:.
|   leaky_relu_rest.py
|   linear_test.py
|   MultiLayerNN.py
|   README.md
|   relu_test.py
|   requirements.txt
|   sigmoid_test.py
|   __init__.py
|
+---activation
|   |   ActivationFunction.py // Base class for activation functions abstract class
|   |   LeakyReLU.py
|   |   Linear.py
|   |   ReLU.py
|   |   Sigmoid.py
|   |   __init__.py
|   |
|   \---__pycache__
|           ActivationFunction.cpython-38.pyc
|           LeakyReLU.cpython-38.pyc
|           Linear.cpython-38.pyc
|           ReLU.cpython-38.pyc
|           ReLUM.cpython-38.pyc
|           Sigmoid.cpython-38.pyc
|           __init__.cpython-38.pyc
|
+---evaluation
|   |   ClassificationEvaluator.py
|   |   __init__.py
|   |
|   \---__pycache__
|           ClassificationEvaluator.cpython-38.pyc
|           __init__.cpython-38.pyc
|
\---__pycache__
        MultiLayerNN.cpython-38.pyc
```

**I explained every class in the video if the comments in code source aren't enough**

## Running the project

- Clone the repo
- Install dependencies in requirement.txt
- Open project in conda enviroment using Anaconda Vscode
- Run the project
