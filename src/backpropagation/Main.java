package backpropagation;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static final double LEARNING_RATE = 0.7;
    public static final double MOMENTUM = 0.3;

    public static void main(String[] args) {
        partA();
        partB();
    }

    public static void partA() {

        double[] biases1 = parseSingleColumn("res/partAData/b1.csv");
        double[] biases2 = parseSingleColumn("res/partAData/b2.csv");
        double[][] weights1 = parseMultiColumn("res/partAData/w1.csv");
        double[][] weights2 = parseMultiColumn("res/partAData/w2.csv");

        double[][] crossData = parseMultiColumn("res/partAData/cross_data.csv");

        Network network = new Network(LEARNING_RATE, MOMENTUM);

        network.addLayer(weights1, biases1);
        network.addLayer(weights2, biases2);

        System.out.println("\nPart A\n");

        trainAndPrintNetwork(network, crossData);

    }

    public static void partB() {

        int inputSize = 4;
        int hiddenSize = 10;
        int outputSize = 2;

        double[] biases1 = randomBiasData(hiddenSize);
        double[] biases2 = randomBiasData(outputSize);

        double[][] weights1 = randomWeightData(hiddenSize, inputSize);
        double[][] weights2 = randomWeightData(outputSize, hiddenSize);

        double[][] gaussian = parseGaussians("res/Two_Class_FourDGaussians500.txt");

        Network network = new Network(LEARNING_RATE, MOMENTUM);

        network.addLayer(weights1, biases1);
        network.addLayer(weights2, biases2);

        System.out.println("\nPart B\n");
        trainAndPrintNetwork(network, gaussian);

    }

    public static double[] parseSingleColumn(String filename) {

        List<Double> biasesList = new ArrayList<>();
        double[] biases;

        try (CSVParser parser = CSVParser.parse(new FileReader(filename), CSVFormat.EXCEL)) {
            for (CSVRecord record : parser) {
                biasesList.add(Double.parseDouble(record.get(0)));
            }
        } catch (IOException ex) {
            return null;
        } finally {

            biases = new double[biasesList.size()];

            for (int i = 0; i < biasesList.size(); i++) {
                biases[i] = biasesList.get(i);
            }

        }

        return biases;

    }

    public static double[][] parseMultiColumn(String filename) {

        List<List<Double>> weightsList = new ArrayList<>();

        double[][] weights;

        try (CSVParser parser = CSVParser.parse(new FileReader(filename), CSVFormat.EXCEL)) {

            for (CSVRecord record : parser) {

                List<Double> newWeights = new ArrayList<>();

                for (String value : record) {
                    newWeights.add(Double.parseDouble(value));
                }

                weightsList.add(newWeights);

            }

        } catch (IOException ex) {
            return null;
        } finally {

            weights = new double[weightsList.size()][weightsList.get(0).size()];

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    weights[i][j] = weightsList.get(i).get(j);
                }
            }

        }

        return weights;

    }

    public static double[][] parseGaussians(String filename) {

        List<List<Double>> inputList = new ArrayList<>();

        double[][] inputs = null;

        try (Scanner scanner = new Scanner(new File(filename))) {
            while (scanner.hasNext()) {

                List list = new ArrayList();

                for (int i = 0; i < 5; i++) {
                    list.add(scanner.nextDouble());
                }

                inputList.add(list);

            }

            inputs = new double[inputList.size()][inputList.get(0).size()];

            for (int i = 0; i < inputs.length; i++) {
                for (int j = 0; j < inputs[0].length; j++) {
                    inputs[i][j] = inputList.get(i).get(j);
                }
            }

            return inputs;

        } catch (FileNotFoundException e) {
            return null;
        }

    }

    public static double[] randomBiasData(int size) {

        double biases[] = new double[size];

        for (int i = 0; i < biases.length; i++) {
            biases[i] = Math.random();
        }

        return biases;

    }


    public static double[][] randomWeightData(int numberOfNeurons, int numberOfInputs) {

        double weights[][] = new double[numberOfNeurons][numberOfInputs];

        for (int i = 0; i < numberOfNeurons; i++) {
            weights[i] = randomBiasData(numberOfInputs);
        }

        return weights;

    }

    public static void trainAndPrintNetwork(Network network, double[][] data) {

        double averageSumOfSquaredErrors = 0;
        double previousSumOfSquaredErrors = 0;
        double changeInError = 0;

        do {
            previousSumOfSquaredErrors = averageSumOfSquaredErrors;
            network.printNetwork();
            averageSumOfSquaredErrors = network.train(data);
            changeInError = Math.abs(previousSumOfSquaredErrors - averageSumOfSquaredErrors);
        } while (changeInError > 0.001);

        // final network
        network.printNetwork();

    }

}
