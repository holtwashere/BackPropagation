package backpropagation;

import java.util.ArrayList;

public class Network {

    public double learningRate;
    public double momentum;

    private ArrayList<Layer> layers;

    public Network(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.layers = new ArrayList<>();
    }

    public void addLayer(double weights[][], double biases[]) {
        layers.add(new Layer(weights, biases));
    }

    public double[] process(double inputs[]) {

        double[] outputs = null;

        for (Layer layer: layers) {
            outputs = layer.process(inputs);
            inputs = outputs;
        }

        return outputs;

    }

    public double train(double inputs[][]) {

        double[] outputs = null;
        double totalSumOfSquaredErrors = 0;

        for (double[] input : inputs) {

            outputs = process(input);

            // Get outer layer
            Layer currentLayer = layers.get(layers.size() - 1);
            Layer nextLayer;

            // check outputs to see if there's a mistake
            for (int i = 0; i < outputs.length; i++) {

                double actual = outputs[i];
                double expected = input[input.length - 1];
                double error = actual - expected;

                totalSumOfSquaredErrors += error * error;
                Neuron neuron = currentLayer.neurons.get(i);
                neuron.setLocalGradient(error, learningRate, momentum);

            }

            // going backwards through the layers!
            for (int i = layers.size() - 2; i >= 0; i--) {

                nextLayer = currentLayer;
                currentLayer = layers.get(i);

                //  cycle through neurons in layer
                for (int j = 0; j < currentLayer.size; j++) {

                    Neuron currentNeuron = currentLayer.neurons.get(j);
                    double error = 0;

                    for (Neuron nextNeuron : nextLayer.neurons) {
                        error += (currentNeuron.getLocalGradient() * nextNeuron.getWeight(j));
                    }

                    currentNeuron.setLocalGradient(error, learningRate, momentum);

                }

            }

        }

        // return the average
        return totalSumOfSquaredErrors / inputs.length;

    }

}
