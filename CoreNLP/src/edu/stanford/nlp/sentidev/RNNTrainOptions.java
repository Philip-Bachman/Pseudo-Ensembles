package edu.stanford.nlp.sentidev;

import java.io.Serializable;

public class RNNTrainOptions implements Serializable {
	public int batchSize = 27;

	/** Number of times through all the trees */
	public int epochs = 400;

	public int debugOutputEpochs = 10;

	public int maxTrainTimeSeconds = 60 * 60 * 120;

	public double learningRate = 0.01;

	public double scalingForInit = 1.0;

	private double[] classWeights = null;

	/**
	 * The classWeights can be passed in as a comma separated list of
	 * weights using the -classWeights flag.  If the classWeights are
	 * not specified, the value is assumed to be 1.0.  classWeights only
	 * apply at train time; we do not weight the classes at all during
	 * test time.
	 */
	public double getClassWeight(int i) {
		if (classWeights == null) {
			if (i == 2) {
				return 1.0;
			} else {
				return 1.0;
			}
		}
		return classWeights[i];
	}

	/** Regularization cost for the transform matrices and tensors */
	public double regTransform = 0.001;

	/** Regularization cost for the classification matrices */
	public double regClassification = 0.0001;

	/** Regularization cost for the word vectors */
	public double regWordVector = 0.0001;

	/** Noise to add to parameters during training */
	public double regParamNoise = 0.0;

	/**
	 * regWordDrop gives the per-word drop rate (actually swap to *UNK* rate).
	 */
	public double regWordDrop = 0.0;

	/**
	 * regActNoise gives the noise to apply to activations in forward prop.
	 */
	public double regActNoise = 0.0;

	/**
	 * The value to set the learning rate for each parameter when initializing adagrad.
	 */
	public double adagradResetWeight = 0.0;

	/** 
	 * How many epochs between resets of the adagrad learning rates.
	 * Set to 0 to never reset.
	 */
	public int adagradResetFrequency = 1;

	@Override
	public String toString() {
		StringBuilder result = new StringBuilder();
		result.append("TRAIN OPTIONS\n");
		result.append("batchSize=" + batchSize + "\n");
		result.append("epochs=" + epochs + "\n");
		result.append("debugOutputEpochs=" + debugOutputEpochs + "\n");
		result.append("maxTrainTimeSeconds=" + maxTrainTimeSeconds + "\n");
		result.append("learningRate=" + learningRate + "\n");
		result.append("scalingForInit=" + scalingForInit + "\n");
		if (classWeights == null) {
			result.append("classWeights=null\n");
		} else {
			result.append("classWeights=");
			result.append(classWeights[0]);
			for (int i = 1; i < classWeights.length; ++i) {
				result.append("," + classWeights[i]);
			}
			result.append("\n");
		}
		result.append("regTransform=" + regTransform + "\n");
		result.append("regClassification=" + regClassification + "\n");
		result.append("regWordVector=" + regWordVector + "\n");
		result.append("regParamNoise=" + regParamNoise + "\n");
		result.append("regActNoise=" + regActNoise + "\n");
		result.append("regWordDrop=" + regWordDrop + "\n");
		result.append("adagradResetWeight=" + adagradResetWeight + "\n");
		result.append("adagradResetFrequency=" + adagradResetFrequency + "\n");
		return result.toString();
	}

	public int setOption(String[] args, int argIndex) {
		if (args[argIndex].equalsIgnoreCase("-batchSize")) {
			batchSize = Integer.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-epochs")) {
			epochs = Integer.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-debugOutputEpochs")) {
			debugOutputEpochs = Integer.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-maxTrainTimeSeconds")) {
			maxTrainTimeSeconds = Integer.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-learningRate")) {
			learningRate = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-scalingForInit")) {
			scalingForInit = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-regTransform")) {
			regTransform = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-regClassification")) {
			regClassification = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-regWordVector")) {
			regWordVector = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-regParamNoise")) {
			regParamNoise = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-regActNoise")) {
			regActNoise = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-regWordDrop")) {
			regWordDrop = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-adagradResetWeight")) {
			adagradResetWeight = Double.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-adagradResetFrequency")) {
			adagradResetFrequency = Integer.valueOf(args[argIndex + 1]);
			return argIndex + 2;
		} else if (args[argIndex].equalsIgnoreCase("-classWeights")) {
			String classWeightString = args[argIndex + 1];
			String[] pieces = classWeightString.split(",");
			classWeights = new double[pieces.length];
			for (int i = 0; i < pieces.length; ++i) {
				classWeights[i] = Double.valueOf(pieces[i]);
			}
			return argIndex + 2;
		} else {
			return argIndex;
		}
	}

	private static final long serialVersionUID = 1;
}
