package edu.stanford.nlp.sentidev;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Map;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.neural.SimpleTensor;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.TwoDimensionalSet;

public class SentimentTraining {

	private static final NumberFormat NF = new DecimalFormat("0.00");
	private static final NumberFormat LF = new DecimalFormat("0.000000");
	private static final NumberFormat FILENAME = new DecimalFormat("0000");

	public static double executeOneTrainingBatch(SentimentModel model, List<Tree> trainingBatch, 
			int epoch, double lRate, double[] adaSGS) {
		// Initialize a gradient/cost computer for this minibatch
		SentimentCostAndGradient gcFunc = new SentimentCostAndGradient(model, trainingBatch);

		// Adagrad parameters...
		double eps = 1e-3;

		// Compute gradient at "lookahead" point, to get the NAG update
		double[] theta = model.paramsToVector();
		double[] grad = gcFunc.derivativeAt(theta);
		double currCost = gcFunc.valueAt(theta);

		// Update the parameters using adagrad
		double adaStep = 0.0;
		for (int i=0; i < theta.length; i++) {
			// Use standard adagrad for awhile
			adaSGS[i] = adaSGS[i] + grad[i]*grad[i];
			adaStep = lRate / (Math.sqrt(adaSGS[i]) + eps);
			theta[i] = theta[i] - (adaStep * grad[i]);
		}

		// Write updated parameters to the model
		model.vectorToParams(theta);    
		return currCost;
	}

	public static void train(SentimentModel model, String modelPath, List<Tree> trainingTrees, List<Tree> devTrees) {
		Timing timing = new Timing();
		long maxTrainTimeMillis = model.op.trainOptions.maxTrainTimeSeconds * 1000;
		int debugCycle = 0;
		int maxEpochs = model.op.trainOptions.epochs;
		int adaResetEpochs = model.op.trainOptions.adagradResetFrequency;
		int outputEpochs = model.op.trainOptions.debugOutputEpochs;
		double adaResetWeight = model.op.trainOptions.adagradResetWeight;
		double bestAccuracy = 0.0;
		double epochCost = 0.0;
		double currCost = 0.0;
		double miniCost = 0.0;
		double lRate = model.op.trainOptions.learningRate;

		// train using AdaGrad
		double[] adaSGS = new double[model.totalParamSize()];
		Arrays.fill(adaSGS, adaResetWeight);

		// If using ReLU, set initial word vectors to be ReLU-friendly
		Map<String, SimpleMatrix> wordVectors = model.wordVectors;

		int numBatches = trainingTrees.size() / model.op.trainOptions.batchSize + 1;
		System.out.println("Training on " + trainingTrees.size() + " trees in " + numBatches + " batches");
		System.out.println("Times through each training batch: " + maxEpochs);
		for (int epoch = 0; epoch < maxEpochs; ++epoch) {
			System.out.println("======================================");
			System.out.println("Starting epoch " + epoch);

			// set switch for word vector training. train word vectors only after
			// some epoch, or for all epochs if words were randomly initialized
			if (epoch >= 10 || model.op.randomWordVectors) {
				model.op.trainWords = true;
				System.out.println("-- Training word vectors...");
			} else {
				model.op.trainWords = false;
				System.out.println("-- Not training word vectors...");
			}

			// Reset adagrad weights every once in a while. Eventually never reset.
			if (epoch < 250 && adaResetEpochs > 0 && (epoch % adaResetEpochs) == 0) {
				System.out.println("Resetting adagrad weights to " + adaResetWeight);
				Arrays.fill(adaSGS, adaResetWeight);
			}
			if (epoch > 250) {
				lRate = lRate * 0.99;
			}

			///////////////////////////////////////////////////////
			// Process the minibatches of phrases for this epoch //
			///////////////////////////////////////////////////////
			List<Tree> shuffledSentences = Generics.newArrayList(trainingTrees);
			Collections.shuffle(shuffledSentences, model.rand);
			epochCost = 0.0;
			miniCost = 0.0;
			for (int batch = 0; batch < numBatches; ++batch) {
				// Get start and end indices for the current minibatch
				int startTree = batch * model.op.trainOptions.batchSize;
				int endTree = (batch + 1) * model.op.trainOptions.batchSize;
				if (endTree + model.op.trainOptions.batchSize > shuffledSentences.size()) {
					endTree = shuffledSentences.size();
				}
				// Do feedforward and backprop on the current minibatch
				currCost = executeOneTrainingBatch(model, shuffledSentences.subList(startTree, endTree), epoch, lRate, adaSGS);
				epochCost += currCost;
				miniCost += currCost;
				// Output a few diagnostics for this minibatch
				long totalElapsed = timing.report();
				if ((batch % 10) == 0) {
					miniCost = miniCost / 10.0;
					System.out.println("Epoch " + epoch + " batch " + batch + "; cost: " + NF.format(miniCost) + " total time: " + totalElapsed + " ms");
					miniCost = 0.0;
				}
			}

			// Display total cost for this epoch
			System.out.println("============================================================");
			System.out.println("EPOCH COST: " + NF.format(epochCost));
			System.out.println("============================================================");
			if ((epoch == (maxEpochs-1)) || ((epoch % outputEpochs) == 0)) {
				// Occasionally check model performance on a validation set...
				double score = 0.0;
				if (devTrees != null) {
					Evaluate eval = new Evaluate(model);
					// Evaluate model performance on a subsample of training data
					eval.eval(shuffledSentences.subList(0, 1000));
					eval.printSummary();
					eval.reset();
					// Evaluate model performance on the dev set
					eval.eval(devTrees);
					eval.printSummary();
					score = eval.exactNodeAccuracy() * 100.0;
				}
				// output an intermediate model
				if (modelPath != null) {
					String tempPath = modelPath;
					if (modelPath.endsWith(".ser.gz")) {
						tempPath = modelPath.substring(0, modelPath.length() - 7) + "-" + FILENAME.format(debugCycle) + "-" + NF.format(score) + ".ser.gz";
					} else if (modelPath.endsWith(".gz")) {
						tempPath = modelPath.substring(0, modelPath.length() - 3) + "-" + FILENAME.format(debugCycle) + "-" + NF.format(score) + ".gz";
					} else {
						tempPath = modelPath.substring(0, modelPath.length() - 3) + "-" + FILENAME.format(debugCycle) + "-" + NF.format(score);
					}
					model.saveSerialized(tempPath);
				}
				++debugCycle;
			}
		}    
	}

	public static boolean runGradientCheck(SentimentModel model, List<Tree> trees) {
		SentimentCostAndGradient gcFunc = new SentimentCostAndGradient(model, trees);
		return gcFunc.gradientCheck(model.totalParamSize(), 50, model.paramsToVector());    
	}

	public static void main(String[] args) {
		RNNOptions op = new RNNOptions();

		String trainPath = "sentimentTreesDebug.txt";
		String devPath = null;

		boolean runGradientCheck = false;
		boolean runTraining = false;

		boolean filterUnknown = false;

		String modelPath = null;

		for (int argIndex = 0; argIndex < args.length; ) {
			if (args[argIndex].equalsIgnoreCase("-train")) {
				runTraining = true;
				argIndex++;
			} else if (args[argIndex].equalsIgnoreCase("-gradientcheck")) {
				runGradientCheck = true;
				argIndex++;
			} else if (args[argIndex].equalsIgnoreCase("-trainpath")) {
				trainPath = args[argIndex + 1];
				argIndex += 2;
			} else if (args[argIndex].equalsIgnoreCase("-devpath")) {
				devPath = args[argIndex + 1];
				argIndex += 2;
			} else if (args[argIndex].equalsIgnoreCase("-model")) {
				modelPath = args[argIndex + 1];
				argIndex += 2;
			} else if (args[argIndex].equalsIgnoreCase("-filterUnknown")) {
				filterUnknown = true;
				argIndex++;
			} else {
				int newArgIndex = op.setOption(args, argIndex);
				if (newArgIndex == argIndex) {
					throw new IllegalArgumentException("Unknown argument " + args[argIndex]);
				}
				argIndex = newArgIndex;
			}
		}

		// read in the trees
		List<Tree> trainingTrees = SentimentUtils.readTreesWithGoldLabels(trainPath);
		System.out.println("Read in " + trainingTrees.size() + " training trees");
		if (filterUnknown) {
			trainingTrees = SentimentUtils.filterUnknownRoots(trainingTrees);
			System.out.println("Filtered training trees: " + trainingTrees.size());
		}

		List<Tree> devTrees = null;
		if (devPath != null) {
			devTrees = SentimentUtils.readTreesWithGoldLabels(devPath);
			System.out.println("Read in " + devTrees.size() + " dev trees");
			if (filterUnknown) {
				devTrees = SentimentUtils.filterUnknownRoots(devTrees);
				System.out.println("Filtered dev trees: " + devTrees.size());
			}
		}

		// TODO: binarize the trees, then collapse the unary chains.
		// Collapsed unary chains always have the label of the top node in
		// the chain
		// Note: the sentiment training data already has this done.
		// However, when we handle trees given to us from the Stanford Parser,
		// we will have to perform this step

		// build an unitialized SentimentModel from the binary productions
		System.out.println("Sentiment model options:\n" + op);
		SentimentModel model = new SentimentModel(op, trainingTrees);

		// TODO: need to handle unk rules somehow... at test time the tree
		// structures might have something that we never saw at training
		// time.  for example, we could put a threshold on all of the
		// rules at training time and anything that doesn't meet that
		// threshold goes into the unk.  perhaps we could also use some
		// component of the accepted training rules to build up the "unk"
		// parameter in case there are no rules that don't meet the
		// threshold

		if (runGradientCheck) {
			runGradientCheck(model, trainingTrees);
		}

		if (runTraining) {
			train(model, modelPath, trainingTrees, devTrees);
			model.saveSerialized(modelPath);
		}
	}

}
