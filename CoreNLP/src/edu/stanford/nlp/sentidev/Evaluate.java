package edu.stanford.nlp.sentidev;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.StringUtils;

public class Evaluate {
	final SentimentCostAndGradient cag;
	final SentimentModel model;

	final int[][] equivalenceClasses;
	final String[] equivalenceClassNames;

	int labelsCorrect;
	int labelsIncorrect;

	// the matrix will be [gold][predicted]
	int[][] labelConfusion;

	int rootLabelsCorrect;
	int rootLabelsIncorrect;

	int[][] rootLabelConfusion;

	IntCounter<Integer> lengthLabelsCorrect;
	IntCounter<Integer> lengthLabelsIncorrect;

	// Stuff for counting binarized label accuracy
	int binLabelsCorrect;
	int binLabelsIncorrect;
	int binRootLabelsCorrect;
	int binRootLabelsIncorrect;

	private static final NumberFormat NF = new DecimalFormat("0.000000");

	public Evaluate(SentimentModel model) {
		this.model = model;
		this.cag = new SentimentCostAndGradient(model, null);
		this.equivalenceClasses = model.op.equivalenceClasses;
		this.equivalenceClassNames = model.op.equivalenceClassNames;

		reset();
	}

	public void reset() {
		labelsCorrect = 0;
		labelsIncorrect = 0;
		labelConfusion = new int[model.op.numClasses][model.op.numClasses];

		rootLabelsCorrect = 0;
		rootLabelsIncorrect = 0;
		rootLabelConfusion = new int[model.op.numClasses][model.op.numClasses];

		binLabelsCorrect = 0;
		binLabelsIncorrect = 0;
		binRootLabelsCorrect = 0;
		binRootLabelsIncorrect = 0;

		lengthLabelsCorrect = new IntCounter<Integer>();
		lengthLabelsIncorrect = new IntCounter<Integer>();
	}

	public void eval(List<Tree> trees) {
		for (Tree tree : trees) {
			eval(tree);
		}
	}

	public void eval(Tree tree) {
		if (model.useDropout) {
			// Multiprop for multiple sampled "child models"
			cag.multiPropagateTree(tree, 64);
		} else {
			// Multiprop for the single true "parent model"
			cag.multiPropagateTree(tree, 1);
		}
		countTree(tree);
		countRoot(tree);
		countLengthAccuracy(tree);
	}

	private int countLengthAccuracy(Tree tree) {
		if (tree.isLeaf()) {
			return 0;
		}
		Integer gold = RNNCoreAnnotations.getGoldClass(tree);
		Integer predicted = RNNCoreAnnotations.getPredictedClass(tree);
		int length;
		if (tree.isPreTerminal()) {
			length = 1;
		} else {
			length = 0;
			for (Tree child : tree.children()) {
				length += countLengthAccuracy(child);
			}
		}
		if (gold >= 0) {
			if (gold.equals(predicted)) {
				lengthLabelsCorrect.incrementCount(length);
			} else {
				lengthLabelsIncorrect.incrementCount(length);
			}
		}
		return length;
	}

	private void countTree(Tree tree) {
		if (tree.isLeaf()) {
			return;
		}
		for (Tree child : tree.children()) {
			countTree(child);
		}
		int gold = RNNCoreAnnotations.getGoldClass(tree) + 0;
		int predicted = RNNCoreAnnotations.getPredictedClass(tree) + 0;
		// Check fine-grained classification
		if (gold >= 0) {
			if (gold == predicted) {
				labelsCorrect++;
			} else {
				labelsIncorrect++;
			}
			labelConfusion[gold][predicted]++;
		}
		// Check binary classification, don't count neutral-classed points
		int binPred = RNNCoreAnnotations.getBinaryPrediction(tree) + 0;
		if (gold != 2) {
			int binGold = 2;
			if (gold < 2) {
				binGold = 0;
			} else if (gold > 2) {
				binGold = 1;
			}
			if (binPred == binGold) {
				binLabelsCorrect++;
			} else {
				binLabelsIncorrect++;
			}
		}
	}

	private void countRoot(Tree tree) {
		int gold = RNNCoreAnnotations.getGoldClass(tree) + 0;
		int predicted = RNNCoreAnnotations.getPredictedClass(tree) + 0;
		if (gold >= 0) {
			if (gold == predicted) {
				rootLabelsCorrect++;
			} else {
				rootLabelsIncorrect++;
			}
			rootLabelConfusion[gold][predicted]++;
		}
		// Check binary classification, don't count neutral-classed points
		int binPred = RNNCoreAnnotations.getBinaryPrediction(tree) + 0;
		if (gold != 2) {
			int binGold = 2;
			if (gold < 2) {
				binGold = 0;
			} else if (gold > 2) {
				binGold = 1;
			}
			if (binPred == binGold) {
				binRootLabelsCorrect++;
			} else {
				binRootLabelsIncorrect++;
			}
		}
	}

	public double exactNodeAccuracy() {
		return (double) labelsCorrect / ((double) (labelsCorrect + labelsIncorrect));
	}

	public double exactRootAccuracy() {
		return (double) rootLabelsCorrect / ((double) (rootLabelsCorrect + rootLabelsIncorrect));
	}

	public Counter<Integer> lengthAccuracies() {
		Set<Integer> keys = Generics.newHashSet();
		keys.addAll(lengthLabelsCorrect.keySet());
		keys.addAll(lengthLabelsIncorrect.keySet());

		Counter<Integer> results = new ClassicCounter<Integer>();
		for (Integer key : keys) {
			results.setCount(key, lengthLabelsCorrect.getCount(key) / (lengthLabelsCorrect.getCount(key) + lengthLabelsIncorrect.getCount(key)));
		}
		return results;
	}

	public void printLengthAccuracies() {
		Counter<Integer> accuracies = lengthAccuracies();
		Set<Integer> keys = Generics.newTreeSet();
		keys.addAll(accuracies.keySet());
		System.out.println("Label accuracy at various lengths:");
		for (Integer key : keys) {
			System.out.println(StringUtils.padLeft(Integer.toString(key), 4) + ": " + NF.format(accuracies.getCount(key)));
		}
	}

	private static void printConfusionMatrix(String name, int[][] confusion) {
		System.out.println(name + " confusion matrix: rows are gold label, columns predicted label");
		for (int i = 0; i < confusion.length; ++i) {
			for (int j = 0; j < confusion[i].length; ++j) {
				System.out.print(StringUtils.padLeft(confusion[i][j], 10));
			}
			System.out.println();
		}
	}

	private static double[] approxAccuracy(int[][] confusion, int[][] classes) {
		int[] correct = new int[classes.length];
		int[] incorrect = new int[classes.length];
		double[] results = new double[classes.length];
		for (int i = 0; i < classes.length; ++i) {
			for (int j = 0; j < classes[i].length; ++j) {
				for (int k = 0; k < classes[i].length; ++k) {
					correct[i] += confusion[classes[i][j]][classes[i][k]];
				}
			}
			for (int other = 0; other < classes.length; ++other) {
				if (other == i) {
					continue;
				}
				for (int j = 0; j < classes[i].length; ++j) {
					for (int k = 0; k < classes[other].length; ++k) {
						incorrect[i] += confusion[classes[i][j]][classes[other][k]];
					}
				}
			}
			results[i] = ((double) correct[i]) / ((double) (correct[i] + incorrect[i]));
		}
		return results;
	}

	private static double approxCombinedAccuracy(int[][] confusion, int[][] classes) {
		int correct = 0;
		int incorrect = 0;
		for (int i = 0; i < classes.length; ++i) {
			for (int j = 0; j < classes[i].length; ++j) {
				for (int k = 0; k < classes[i].length; ++k) {
					correct += confusion[classes[i][j]][classes[i][k]];
				}
			}
			for (int other = 0; other < classes.length; ++other) {
				if (other == i) {
					continue;
				}
				for (int j = 0; j < classes[i].length; ++j) {
					for (int k = 0; k < classes[other].length; ++k) {
						incorrect += confusion[classes[i][j]][classes[other][k]];
					}
				}
			}
		}
		return ((double) correct) / ((double) (correct + incorrect));
	}

	public void printSummary() {
		System.out.println("EVALUATION SUMMARY");
		System.out.println("Tested " + (labelsCorrect + labelsIncorrect) + " labels");
		System.out.println("  " + labelsCorrect + " correct");
		System.out.println("  " + labelsIncorrect + " incorrect");
		System.out.println("  " + NF.format(exactNodeAccuracy()) + " accuracy");
		System.out.println("Tested " + (rootLabelsCorrect + rootLabelsIncorrect) + " roots");
		System.out.println("  " + rootLabelsCorrect + " correct");
		System.out.println("  " + rootLabelsIncorrect + " incorrect");
		System.out.println("  " + NF.format(exactRootAccuracy()) + " accuracy");

		printConfusionMatrix("Label", labelConfusion);
		printConfusionMatrix("Root label", rootLabelConfusion);

		if (equivalenceClasses != null && equivalenceClassNames != null) {
			double[] approxLabelAccuracy = approxAccuracy(labelConfusion, equivalenceClasses);
			for (int i = 0; i < equivalenceClassNames.length; ++i) {
				System.out.println("Approximate " + equivalenceClassNames[i] + " label accuracy: " + NF.format(approxLabelAccuracy[i]));
			}
			System.out.println("Combined approximate label accuracy: " + NF.format(approxCombinedAccuracy(labelConfusion, equivalenceClasses)));

			double[] approxRootLabelAccuracy = approxAccuracy(rootLabelConfusion, equivalenceClasses);
			for (int i = 0; i < equivalenceClassNames.length; ++i) {
				System.out.println("Approximate " + equivalenceClassNames[i] + " root label accuracy: " + NF.format(approxRootLabelAccuracy[i]));
			}
			System.out.println("Combined approximate root label accuracy: " + NF.format(approxCombinedAccuracy(rootLabelConfusion, equivalenceClasses)));
		}
		//printLengthAccuracies();
		double binAcc = ((double) binLabelsCorrect) / (binLabelsCorrect + binLabelsIncorrect);
		double binRootAcc = ((double) binRootLabelsCorrect) / (binRootLabelsCorrect + binRootLabelsIncorrect);
		System.out.println("Binary full accuracy: " + NF.format(binAcc));
		System.out.println("Binary root accuracy: " + NF.format(binRootAcc));
	}

	/**
	 * Expected arguments are <code> -model model -treebank treebank </code> <br>
	 *
	 * For example <br>
	 * <code> 
	 *  java edu.stanford.nlp.sentiment.Evaluate 
	 *   edu/stanford/nlp/models/sentiment/sentiment.ser.gz 
	 *   /u/nlp/data/sentiment/trees/dev.txt
	 * </code>
	 */
	public static void main(String[] args) {
		String modelPath = null;
		String treePath = null;
		boolean filterUnknown = false;

		for (int argIndex = 0; argIndex < args.length; ) {
			if (args[argIndex].equalsIgnoreCase("-model")) {
				modelPath = args[argIndex + 1];
				argIndex += 2;
			} else if (args[argIndex].equalsIgnoreCase("-treebank")) {
				treePath = args[argIndex + 1];
				argIndex += 2;
			} else if (args[argIndex].equalsIgnoreCase("-filterUnknown")) {
				filterUnknown = true;
				argIndex++;
			} else {
				System.out.println("Unknown argument " + args[argIndex]);
				System.exit(2);
			}
		}

		List<Tree> trees = SentimentUtils.readTreesWithGoldLabels(treePath);
		if (filterUnknown) {
			trees = SentimentUtils.filterUnknownRoots(trees);
		}
		SentimentModel model = SentimentModel.loadSerialized(modelPath);

		Evaluate eval = new Evaluate(model);
		eval.eval(trees);
		eval.printSummary();
	}
}
