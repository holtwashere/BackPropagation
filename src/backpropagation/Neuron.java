package backpropagation;

public class Neuron {

    private double[] weights; // input weights
    private double bias;
    private double[] previousWeights;

    private double[] inputs;
    private double output;

    public double localGradient;

    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
        this.previousWeights = new double[weights.length];
    }

    public double process(double[] inputs) {

        this.inputs = inputs;

        double sum = 0;

        // Compute Dot Product of inputs and weights
        for (int i = 0; i < inputs.length - 1; i++) {
            sum += inputs[i] * weights[i];
        }

        sum += bias;

        this.output = activation(sum);

        return output;

    }

    public void setLocalGradient(double error, double learningRate, double momentum) {

        localGradient = activationPrime() * error;

        for (int i = 0; i < weights.length; i++) {

            double currentWeight = weights[i];
            double previousWeight = previousWeights[i];

            weights[i] += (momentum * (currentWeight - previousWeight)) + (learningRate * localGradient * inputs[i]);
            previousWeights[i] = currentWeight;

        }

    }

    public double getLocalGradient() {
        return localGradient;
    }

    public double getWeight(int index) {
        return weights[index];
    }

    public double activationPrime() {
        return output * (1 - output);
    }

    public static double activation(double v) {
        return (double) 1 / (1 + Math.exp(-v));
    }

}
