package backpropogation;

import java.util.ArrayList;

public class Layer {

    private ArrayList<Neuron> neurons;
    private int size;

    public Layer(double weights[][], double biases[]) {

        neurons = new ArrayList<>();
        size = weights.length;

        for (int i = 0; i < size; i++) {
            this.neurons.add(new Neuron(weights[i], biases[i]));
        }

    }

    public double[] process(double[] inputs) {

        double[] outputs = new double[size];

        for (int i = 0; i < size; i++) {
            outputs[i] = neurons.get(i).process(inputs);
        }

        return outputs;

    }

}
