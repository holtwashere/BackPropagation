package backpropagation;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static final double LEARNING_RATE = 0.7;
    public static final double MOMENTUM = 0.3;

    public static void main(String[] args) {

        double[] biases1 = parseSingleColumn("res/partAData/b1.csv");
        double[] biases2 = parseSingleColumn("res/partAData/b2.csv");
        double[][] weights1 = parseMultiColumn("res/partAData/w1.csv");
        double[][] weights2 = parseMultiColumn("res/partAData/w2.csv");

        double[][] crossData = parseMultiColumn("res/partAData/cross_data.csv");

        final Network network = new Network(LEARNING_RATE, MOMENTUM);

        network.addLayer(weights1, biases1);
        network.addLayer(weights2, biases2);
        double[] outputs = network.process(crossData);

        for (double output : outputs) {
            System.out.println(output);
        }

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

            int i = 0;
            for (Double bias: biasesList) {
                biases[i] = bias;
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


}
