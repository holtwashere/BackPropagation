package backpropagation;

import java.util.ArrayList;

public class Network {

    private double learningRate;
    private double momentum;

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

        double[] outputs = new double[0];

        for (Layer layer: layers) {
            outputs = layer.process(inputs);
            inputs = outputs;
        }

        return outputs;

    }

    public void train(double inputs[]) {

    }

}
