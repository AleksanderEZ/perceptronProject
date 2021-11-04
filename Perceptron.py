class SingleLayerPerceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.input_data = 0
        self.expected_outputs = 0
        self.input_layer = 0
        self.output_layer = 0
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_input_layer(self, layer):
        self.input_layer = layer

    def add_output_layer(self, layer):
        self.output_layer = layer

    def fit(self, input_data, expected_outputs):
        self.input_data = input_data
        self.expected_outputs = expected_outputs

        self.initializeNetworkWeights()

        for _ in range(self.epochs):
            output = self.predict(self.input_data)
            self.updateNetworkWeights(output)

    def initializeNetworkWeights(self):
        self.input_layer.initializeWeights()
        self.output_layer.initializeWeights()
        for layer in self.layers:
            layer.initializeWeights()

    def predict(self, input):
        self.input_layer.predict(input)
        last_input = 0 #layer->layer->layer
        return self.output_layer.predict(last_input)

    def updateNetworkWeigths(self, output):
        self.layers[-1].updateWeights(output, self.input_data, self.expected_outputs, self.learning_rate)
