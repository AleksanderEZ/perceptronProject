from Perceptron import SingleLayerPerceptron
from InputLayer import InputLayer
from Layer import Layer
from OutputLayer import OutputLayer

learning_rate = 0.1 #[0,1]
epochs = 100

input_vector = ([2, 2], [2, 5], [4, 3], [4, 4], [0, 3], [3, 0], [4, 6], [6, 2])
desired_output = ([1, 1, 1, 1, 0, 0, 0, 0])
testData = ([1, 1], [3, 6], [3, 2], [5, 3])

perceptron = SingleLayerPerceptron(learning_rate, epochs)
perceptron.add_input_layer(InputLayer(4))
perceptron.add_layer(Layer(3))
perceptron.add_output_layer(OutputLayer(1))
perceptron.fit(input_data=input_vector, expected_outputs=desired_output)

print(perceptron.predict(testData))
