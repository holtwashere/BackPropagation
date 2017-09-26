package backpropagation;

public class Neuron {

    private double[] weights; // input weights
    private double bias;

    private double[] inputs;
    private double output;

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public double process(double[] inputs) {

        this.inputs = inputs;

        double sum = 0;

        // Compute Dot Product of inputs and weights
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }

        sum += bias;

        this.output = activation(sum);

        return output;

    }

    public double activationPrime() {
        return output * (1 - output);
    }

    public static double activation(double v) {
        return (double) 1 / (1 + Math.exp(-v));
    }

}
