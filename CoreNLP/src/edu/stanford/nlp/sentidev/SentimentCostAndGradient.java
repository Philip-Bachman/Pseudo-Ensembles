package edu.stanford.nlp.sentidev;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.*;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.neural.NeuralUtils;
import edu.stanford.nlp.neural.SimpleTensor;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.optimization.AbstractCachingDiffFunction;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.TwoDimensionalMap;

// TODO: get rid of the word Sentiment everywhere
public class SentimentCostAndGradient extends AbstractCachingDiffFunction {
	SentimentModel model;
	List<Tree> trainingBatch;
	List<Integer> dropIdx;

	public SentimentCostAndGradient(SentimentModel model, List<Tree> trainingBatch) {
		this.model = model;
		this.trainingBatch = trainingBatch;
		// Create a list of valid indices that will be subsampled for dropout
		this.dropIdx = Generics.newArrayList();
		for (int i=0; i < model.numHid; i++) {
			dropIdx.add(i);
		}
	}

	public int domainDimension() {
		// TODO: cache this for speed?
		return model.totalParamSize();
	}

	private static double sumError(Tree tree) {
		if (tree.isLeaf()) {
			return 0.0;
		} else if (tree.isPreTerminal()) {
			return RNNCoreAnnotations.getPredictionError(tree);
		} else {
			double error = 0.0;
			for (Tree child : tree.children()) {
				error += sumError(child);
			}
			return RNNCoreAnnotations.getPredictionError(tree) + error;
		}
	}

	private int countWords(Tree tree) {
		int subTreeWordCount = 0;
		if (tree.isPreTerminal()) {
			subTreeWordCount = 1;
		} else {
			subTreeWordCount =  countWords(tree.children()[0]) + 
					countWords(tree.children()[1]);
		}
		return subTreeWordCount;
	}

	/**
	 * Returns the index with the highest value in the <code>predictions</code> matrix.
	 * Indexed from 0.
	 */
	public int getPredictedClass(SimpleMatrix predictions) {
		int argmax = 0;
		for (int i = 1; i < predictions.getNumElements(); ++i) {
			if (predictions.get(i) > predictions.get(argmax)) {
				argmax = i;
			}
		}
		return argmax;
	}

	/**
	 * Returns the binarized classification for this point.
	 */
	public int getBinaryPrediction(SimpleMatrix predictions) {
		int argmax = 0;
		for (int i = 1; i < predictions.getNumElements(); ++i) {
			if (i != 2) {
				if (predictions.get(i) > predictions.get(argmax)) {
					argmax = i;
				}
			}
		}
		if (argmax < 2) {
			argmax = 0;
		} else {
			argmax = 1;
		}
		return argmax;
	}

	public void calculate(double[] theta) {
		model.vectorToParams(theta);

		double localValue = 0.0;
		double[] localDerivative = new double[theta.length];

		SimpleMatrix binaryMatrix = model.binaryMatrix;
		SimpleTensor binaryTensor = model.binaryTensor;
		SimpleMatrix classyMatrix = model.classyMatrix;
		Map<String, SimpleMatrix> wordVectors = model.wordVectors;

		// Setup a matrix to hold the derivatives on model.binaryMatrix
		final SimpleMatrix binaryMD = new SimpleMatrix(binaryMatrix.numRows(), binaryMatrix.numCols());
		// Setup a tensor to hold the derivatives on model.binaryTensor
		final SimpleTensor binaryTD = new SimpleTensor(binaryTensor.numRows(), binaryTensor.numCols(), binaryTensor.numSlices());
		// Setup a matrix to hold the derivatives on model.classyMatrix
		final SimpleMatrix classyMD = new SimpleMatrix(classyMatrix.numRows(), classyMatrix.numCols());
		// Setup a map to hold the derivatives on each word vector
		final Map<String, SimpleMatrix> wordVD = Generics.newTreeMap();
		for (Map.Entry<String, SimpleMatrix> entry : wordVectors.entrySet()) {
			int numRows = entry.getValue().numRows();
			int numCols = entry.getValue().numCols();
			wordVD.put(entry.getKey(), new SimpleMatrix(numRows, numCols));
		}

		// Call-out to multithreaded feedforward and backprop method.
		double error = 0.0;
		try {
			error = ffbpTrees(trainingBatch, binaryMD, binaryTD, classyMD, wordVD);
		}
		catch (Exception e) {
			e.printStackTrace();
			throw new AssertionError("Die fast, die hard!");
		}

		// scale the error by the number of sentences so that the effect of
		// regularization is somewhat invariant w.r.t. the training batch size
		double scale = (1.0 / trainingBatch.size());
		value = error * scale;

		// Add regularization terms to the loss/gradient
		double scaledRegBM = model.op.trainOptions.regTransform;
		double scaledRegBT = model.op.trainOptions.regTransform;
		double scaledRegCM = model.op.trainOptions.regClassification;
		double scaledRegWV = model.op.trainOptions.regWordVector;
		if (model.useDropout) {
			// Scale regularization parameters to compensate for droppage
			scaledRegBM = scaledRegBM / 1.0; //4.0;
			scaledRegBT = scaledRegBT / 1.0; //8.0;
			scaledRegCM = scaledRegCM / 1.0; //2.0;
			scaledRegWV = scaledRegWV / 1.0; //2.0;
		}
		value += scaleAndRegularizeMatrix(binaryMD, binaryMatrix, scale, scaledRegBM);
		value += scaleAndRegularizeTensor(binaryTD, binaryTensor, scale, scaledRegBT);
		value += scaleAndRegularizeMatrix(classyMD, classyMatrix, scale, scaledRegCM);
		value += scaleAndRegularizeWords(wordVD, wordVectors, scale, scaledRegWV);
		// clamp word vector gradients to 0 if we aren't training them
		if (model.op.trainWords != true) {
			for (Map.Entry<String, SimpleMatrix> entry : wordVD.entrySet()) {
				wordVD.put(entry.getKey(), entry.getValue().scale(0.0));
			}
		}

		// Record the gradient information for outside use
		List<SimpleMatrix> paramIterator = Generics.newArrayList();
		paramIterator.add(binaryMD);
		paramIterator.add(classyMD);
		derivative = NeuralUtils.paramsToVector(theta.length, paramIterator.iterator(),
				binaryTD.iteratorSimpleMatrix(), wordVD.values().iterator());
	}

	double scaleAndRegularizeMatrix(SimpleMatrix derivatives,
			SimpleMatrix currentMatrix,
			double scale,
			double regCost) {
		derivatives.set(derivatives.scale(scale).plus(currentMatrix.scale(regCost)));
		regCost = 0.5 * (currentMatrix.elementMult(currentMatrix).elementSum() * regCost);
		return regCost;
	}

	double scaleAndRegularizeTensor(SimpleTensor derivatives,
			SimpleTensor currentTensor,
			double scale,
			double regCost) {
		derivatives.set(derivatives.scale(scale).plus(currentTensor.scale(regCost)));
		regCost = 0.5 * (currentTensor.elementMult(currentTensor).elementSum() * regCost);
		return regCost;
	}

	double scaleAndRegularizeWords(Map<String, SimpleMatrix> derivatives,
			Map<String, SimpleMatrix> currentWords,
			double scale,
			double regCost) {
		double cost = 0.0; // the regularization cost
		for (Map.Entry<String, SimpleMatrix> entry : currentWords.entrySet()) {
			SimpleMatrix D = derivatives.get(entry.getKey());
			D = D.scale(scale).plus(entry.getValue().scale(regCost));
			derivatives.put(entry.getKey(), D);
			cost += 0.5 * (entry.getValue().elementMult(entry.getValue()).elementSum() * regCost);
		}
		return cost;
	}

	/////////////////////////////////
	// DEAL WITH THE NON-LINEARITY //
	/////////////////////////////////

	private SimpleMatrix fpropActivation(SimpleMatrix nodePreVector) {
		SimpleMatrix nodeVector = null;
		// Fprop through tanh
		nodeVector = NeuralUtils.elementwiseApplyTanh(nodePreVector);
		return nodeVector;
	}

	private SimpleMatrix bpropActivation(SimpleMatrix nodeGrad,
			SimpleMatrix nodeVector,
			SimpleMatrix nodePreVector) {
		SimpleMatrix nodePreGrad = null;
		// Bprop through tanh
		SimpleMatrix tanhGrad = 
				NeuralUtils.elementwiseApplyTanhDerivative(nodeVector);
		nodePreGrad = nodeGrad.elementMult(tanhGrad);
		return nodePreGrad;
	}


	//////////////////////////////////////////////////////////////////////
	// BACKPROPS THROUGH FULL NODE TRANSFORM, WITH AND WITHOUT DROPPAGE //
	//////////////////////////////////////////////////////////////////////

	/**
	 * This is called at the root of the tree and may need to do some special
	 * stuff to deal with submodel sampling (ie, dropout).
	 */
	private void backpropDerivativesAndError(Tree tree,
			SimpleMatrix binaryMD,
			SimpleTensor binaryTD,
			SimpleMatrix classyMD,
			Map<String, SimpleMatrix> wordVD) {
		// Setup "containers" to track the gradient information for this tree
		SimpleMatrix liveIdx = RNNCoreAnnotations.getLiveIdx(tree);
		int subDim = liveIdx.numRows();
		SimpleMatrix subBinaryMD = new SimpleMatrix(subDim, ((2 * subDim) + 1));
		SimpleTensor subBinaryTD = new SimpleTensor(subDim, subDim, subDim);
		SimpleMatrix subClassyMD = new SimpleMatrix(classyMD.numRows(), (subDim + 1));
		SimpleMatrix delta = new SimpleMatrix(subDim, 1);
		// Run backprop, and collect gradient information in the appropriately
		// sized containers. Note that wordVector gradient information will be
		// applied directly to the full-sized wordVD (after unslicing).
		backpropDerivativesAndError(tree, subBinaryMD, subBinaryTD, subClassyMD, wordVD, delta);
		// Now, unslice the gradient matrices/tensors and push them back onto the
		// the shared (between threads) full-sized gradient accumulators...
		synchronized(binaryMD) {
			binaryMD.set(binaryMD.plus(unsliceBinaryMatrix(liveIdx, subBinaryMD)));
		}
		synchronized(binaryTD) {
			binaryTD.set(binaryTD.plus(unsliceBinaryTensor(liveIdx, subBinaryTD)));
		}
		synchronized(classyMD) {
			classyMD.set(classyMD.plus(unsliceClassyMatrix(liveIdx, subClassyMD)));
		}
		// Clean up annotations/pointers to submodel info
		cleanTree(tree);
	}

	private void backpropDerivativesAndError(Tree tree,
			SimpleMatrix binaryMD,
			SimpleTensor binaryTD,
			SimpleMatrix classyMD,
			Map<String, SimpleMatrix> wordVD,
			SimpleMatrix deltaUp) {
		///////////////////////////////////////////////////////////////////////////////////////
		// Inputs:                                                                           //
		//   tree: the phrase (sub)tree through which to backprop gradients                  //
		//   binaryMD: gradient accumulator for matrix/linear part of transform              //
		//   binaryTD: gradient accumulator for tensor/bilinear part of transform            //
		//   classyMD: gradient accumulator for classification matrix                        //
		//   wordVD: gradient accumulator for the word embedding vectors                     //
		//   deltaUp: gradients pushed back onto this node's nodeVector from above           //
		///////////////////////////////////////////////////////////////////////////////////////
		if (tree.isLeaf()) {
			return;
		}

		SimpleMatrix nodeVector = RNNCoreAnnotations.getNodeVector(tree);
		SimpleMatrix noisyNodeVector = RNNCoreAnnotations.getNoisyNodeVector(tree);
		SimpleMatrix nodePreVector = RNNCoreAnnotations.getNodePreVector(tree);
		SimpleMatrix liveIdx = RNNCoreAnnotations.getLiveIdx(tree);
		SimpleMatrix binaryMatrix = RNNCoreAnnotations.getBinaryMatrix(tree);
		SimpleTensor binaryTensor = RNNCoreAnnotations.getBinaryTensor(tree);
		SimpleMatrix classyMatrix = RNNCoreAnnotations.getClassyMatrix(tree);
		int subDim = nodeVector.numRows();

		// Compute the loss/gradient on the classifier output for this node
		int goldClass = RNNCoreAnnotations.getGoldClass(tree);
		SimpleMatrix deltaClass = computeDeltaClass(tree, noisyNodeVector, goldClass);

		// Compute and update gradient of prediction loss w.r.t. classyMatrix
		SimpleMatrix localCD = deltaClass.mult(
				NeuralUtils.concatenateWithBias(noisyNodeVector).transpose());
		classyMD.set(classyMD.plus(localCD));

		// Compute gradient on nodeVector due to prediction loss
		SimpleMatrix deltaFromClass = classyMatrix.transpose().mult(deltaClass);
		deltaFromClass = deltaFromClass.extractMatrix(0, subDim, 0, 1);

		// Combine gradient on nodeVector from prediction loss with gradient on
		// nodeVector pushed down by backprop. Then bp through the activation fn.
		SimpleMatrix deltaFull = deltaFromClass.plus(deltaUp);
		deltaFull = bpropActivation(deltaFull, nodeVector, nodePreVector);

		if (tree.isPreTerminal()) { 
			// Here we do backprop through a node which represents a single word,
			// i.e. only the gradient for a word embedding vector will be affected.
			String word = tree.children()[0].label().value();
			word = model.getVocabWord(word);
			// Add the gradient from this node to the grad accumulator for "word"
			synchronized(wordVD){
				wordVD.put(word, wordVD.get(word).plus(unsliceWordVector(liveIdx, deltaFull)));
			}
		} else {
			// Here we do backprop through an internal node of the phrase tree.
			double lWeight = 1.0;
			double rWeight = 1.0;
			if (model.op.wordWeight) {
				double lWC = (double) countWords(tree.children()[0]);
				double rWC = (double) countWords(tree.children()[1]);
				lWeight = lWC / (lWC + rWC);
				rWeight = rWC / (lWC + rWC);
			}
			SimpleMatrix lcVec = RNNCoreAnnotations.getNoisyNodeVector(tree.children()[0]).scale(lWeight);
			SimpleMatrix rcVec = RNNCoreAnnotations.getNoisyNodeVector(tree.children()[1]).scale(rWeight);
			SimpleMatrix jcVec = NeuralUtils.concatenateWithBias(lcVec, rcVec);
			// Compute loss gradients with respect to the parameters of the basic
			// linear transform part of this node.
			SimpleMatrix W_df = deltaFull.mult(jcVec.transpose());
			binaryMD.set(binaryMD.plus(W_df));
			SimpleMatrix deltaDown;
			if (model.op.useTensors) {
				// Compute loss gradients with respect to the parameters of the bilinear
				// transform part of this node.
				SimpleTensor T_df = getTensorGradient(deltaFull, lcVec, rcVec);
				binaryTD.set(binaryTD.plus(T_df));
				// Compute loss gradients through bilinear tensor transform and basic
				// linear transform with respect to the outputs of the child nodes.
				deltaDown = computeTensorDeltaDown(deltaFull, lcVec, rcVec, binaryMatrix, binaryTensor);
			} else {
				// Compute loss gradients through the basic linear transform with
				// respect to the outputs of the child nodes.
				deltaDown = binaryMatrix.transpose().mult(deltaFull);
			}
			SimpleMatrix lDeltaDown = deltaDown.extractMatrix(0, subDim, 0, 1).scale(lWeight);
			SimpleMatrix rDeltaDown = deltaDown.extractMatrix(subDim, (subDim * 2), 0, 1).scale(rWeight);
			backpropDerivativesAndError(tree.children()[0], binaryMD, binaryTD, classyMD, wordVD, lDeltaDown);
			backpropDerivativesAndError(tree.children()[1], binaryMD, binaryTD, classyMD, wordVD, rDeltaDown);
		}
	}

	/**
	 * Compute gradients on the given "predictions", for an observation which
	 * belongs to class indicated by "goldClass".
	 */
	private SimpleMatrix computeDeltaClass(Tree tree, SimpleMatrix inputVector, int goldClass) {

		// If this is a wacky class, DIE HARD!
		if ((goldClass < 0) || (goldClass > 4)) {
			throw new AssertionError("Classes should be in {0, 1, 2, 3, 4}.");
		}

		// Otherwise, setup a binary class indicator vector...
		SimpleMatrix goldLabel = new SimpleMatrix(model.numClasses,1);
		if (goldClass >= 0) {
			goldLabel.set(goldClass, 1.0);
		}

		double nodeWeight = model.op.trainOptions.getClassWeight(goldClass);
		SimpleMatrix classyMatrix = RNNCoreAnnotations.getClassyMatrix(tree);
		SimpleMatrix linePreds = classyMatrix.mult(NeuralUtils.concatenateWithBias(inputVector));

		double error = 0.0;
		SimpleMatrix deltaClass = new SimpleMatrix(5, 1);
		if (model.useBinary) {
			// Compute loss and what-not for binary prediction mode. In binary
			// prediction mode we will only use (and train) the prediction in
			// linePreds[2]. For phrases with true class 0/1 we will try to
			// make this value < 0, and for phrases with true 3/4 we will try
			// to make it > 0. Phrases with class 2 will have neither any loss
			// nor any gradient.
			double binLabel = -1.0;
			if (goldClass > 2){
				binLabel = 1.0;
			}
			double binPred = linePreds.get(2);
			error = Math.log(1.0 + Math.exp(-1.0 * binLabel * binPred));
			double grad = -binLabel / (Math.exp(binLabel * binPred) + 1.0);
			deltaClass = linePreds.scale(0.0);
			deltaClass.set(2, grad);
			if (goldClass == 2) {
				// Don't do anything for phrases in the neutral class
				deltaClass = deltaClass.scale(0.0);
				error = 0.0;
			}
		} else {
			// Compute loss and what-not for fine-grained prediction mode. In
			// fine-grained mode we just apply the usual softmax + cross-ent
			// loss. We'll also add a "shaping" term, which seems to help.
			SimpleMatrix smaxPreds = NeuralUtils.softmax(linePreds);
			error = -Math.log(smaxPreds.get(goldClass,0)) * nodeWeight;
			deltaClass = smaxPreds.minus(goldLabel).scale(nodeWeight);
			// Add a shaped penalty that more closely echoes the relationships
			// among the possible classes (i.e. their "number-line" positions).
			double pT = linePreds.get(goldClass);
			double pF = 0.0;
			double errWeight = 0.0;
			double regParam = 0.2;
			/*
			for (int i=0; i < 5; i++) {
				if (i != goldClass) {
					pF = linePreds.get(i);
					errWeight = regParam * Math.abs(i - goldClass);
					// Do L2 hinge loss and gradient...
					if (pT < (pF + 1.0)) {
						// Compute margin transgression...
						double marTrans = (pF + 1.0) - pT;
						// Compute the corresponding loss and grad...
						error = error + (errWeight * 0.5 * (marTrans * marTrans));
						deltaClass.set(i, (deltaClass.get(i) + (errWeight * marTrans)));
						deltaClass.set(goldClass, (deltaClass.get(goldClass) - (errWeight * marTrans)));
					}
				}
			}
			*/
		}

		// Record the classification-related error for this node
		RNNCoreAnnotations.setPredictionError(tree, error);
		return deltaClass;
	}

	/////////////////////////////////////////////////////////////////////////////
	// FORWARD PROPAGATION THROUGH PHRASE TREE, TO SET VARIOUS VALUES AT EACH  //
	// INTERNAL NODE.                                                          //
	/////////////////////////////////////////////////////////////////////////////

	/**
	 * This is the method to call for assigning labels and node vectors
	 * to the Tree (during training). This method fills in RNNCoreAnnotations
	 * required for performing backprop and computing the classification error
	 * for the feedforward pass through this phrase tree.
	 */
	public void forwardPropagateTree(Tree tree, boolean addNoise) {
		//
		// nodeVector gives the post-activation output of each node of the RNTN
		// phrase tree. Internal nodes produce nodeVector through a transform of
		// nodeVectors of their children, defined by binaryTensor/binaryMatrix.
		// preTerminal nodes produce nodeVector by simply applying the non-linear
		// activation function to the embedding vector for some word.
		//
		SimpleMatrix liveIdx = RNNCoreAnnotations.getLiveIdx(tree);
		SimpleMatrix binaryMatrix = RNNCoreAnnotations.getBinaryMatrix(tree);
		SimpleTensor binaryTensor = RNNCoreAnnotations.getBinaryTensor(tree);
		SimpleMatrix classyMatrix = RNNCoreAnnotations.getClassyMatrix(tree);
		SimpleMatrix nodePreVector = null;
		SimpleMatrix nodeVector = null;
		SimpleMatrix noisyNodeVector = null;
		SimpleMatrix noise = null;
		double noiseStd = 0.0;

		if (tree.isLeaf()) {
			// We do nothing for the leaves.  The preterminals will
			// calculate the classification for this word/tag.
			throw new AssertionError("We should not have reached leaves in forwardPropagate");
		} else if (tree.isPreTerminal()) {
			// Get the word embedding vector for the child node, and transform it
			// by applying some non-linearity.
			String word = tree.children()[0].label().value();
			if (model.op.trainOptions.regWordDrop > 1e-4) {
				// Drop words at random...
				if (model.rand.nextFloat() < model.op.trainOptions.regWordDrop) {
					word = model.UNKNOWN_WORD;
				}
			}
			SimpleMatrix wordVector = model.getWordVector(word);
			wordVector = sliceWordVector(liveIdx, wordVector);
			if (addNoise == true) {
				// Add noise to word vector if so desired...
				noiseStd = model.op.trainOptions.regParamNoise;
				noise = NeuralUtils.randomGaussian(wordVector.numRows(), 1, model.rand);
				nodePreVector = wordVector.plus(noise.scale(noiseStd));
			} else {
				nodePreVector = wordVector.scale(1.0);
			}
		} else if (tree.children().length == 1) {
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			// Compute weights for each child, based on "words-in-subtree"
			double lWeight = 1.0;
			double rWeight = 1.0;
			if (model.op.wordWeight) {
				double lWC = (double) countWords(tree.children()[0]);
				double rWC = (double) countWords(tree.children()[1]);
				lWeight = lWC / (lWC + rWC);
				rWeight = rWC / (lWC + rWC);
			}
			// fprop child subtrees, to make their results available to this node.
			forwardPropagateTree(tree.children()[0], addNoise);
			forwardPropagateTree(tree.children()[1], addNoise);
			// Compute RNTN fprop through this node.
			SimpleMatrix lcVec = RNNCoreAnnotations.getNoisyNodeVector(tree.children()[0]).scale(lWeight);
			SimpleMatrix rcVec = RNNCoreAnnotations.getNoisyNodeVector(tree.children()[1]).scale(rWeight);
			SimpleMatrix jcVec = NeuralUtils.concatenateWithBias(lcVec, rcVec);
			if (model.op.useTensors) {
				SimpleMatrix tensorOut = binaryTensor.myBilinearProducts(lcVec, rcVec);
				nodePreVector = binaryMatrix.mult(jcVec).plus(tensorOut);
			} else {
				nodePreVector = binaryMatrix.mult(jcVec);
			}
		} else {
			throw new AssertionError("Tree not correctly binarized");
		}
		// Apply activation function to nodePreVector to get nodeVector
		nodeVector = fpropActivation(nodePreVector);

		if (addNoise == true) {
			// Add forward propagation noise if requested
			noiseStd = model.op.trainOptions.regActNoise;
			noise = NeuralUtils.randomGaussian(nodeVector.numRows(), 1, model.rand);
			noisyNodeVector = nodeVector.plus(noise.scale(noiseStd));
		} else {
			noisyNodeVector = nodeVector.scale(1.0);
		}

		// Compute classifier outputs/predictions for this node.
		SimpleMatrix linearPreds = classyMatrix.mult(NeuralUtils.concatenateWithBias(nodeVector));
		SimpleMatrix predictions = linearPreds.scale(1.0);
		if (model.useBinary) {
			// When running in binary mode, we only want to use the value in
			// linearPreds[2]. But, we'll move elsewhere, to let the normal
			// prediction code ignore the binary/fine-grained distinction.
			linearPreds.set(0, -linearPreds.get(2));
			linearPreds.set(4, linearPreds.get(2));
			linearPreds.set(1, 0.0);
			linearPreds.set(2, 0.0);
			linearPreds.set(3, 0.0);
			predictions = linearPreds.scale(1.0);
		}

		// Store the values needed to backprop through this node.
		int finPred = getPredictedClass(predictions);
		int binPred = getBinaryPrediction(predictions);
		if (!(tree.label() instanceof CoreLabel)) {
			throw new AssertionError("Expected CoreLabels in the nodes");
		}
		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.Predictions.class, predictions);
		label.set(RNNCoreAnnotations.PredictedClass.class, finPred);
		label.set(RNNCoreAnnotations.BinaryPrediction.class, binPred);
		label.set(RNNCoreAnnotations.NodeVector.class, nodeVector);
		label.set(RNNCoreAnnotations.NodePreVector.class, nodePreVector);
		label.set(RNNCoreAnnotations.NoisyNodeVector.class, noisyNodeVector);
	}

	//////////////////////////////////////////////////////////////
	// BACKPROPS THROUGH TENSOR TRANSFORM FOR PARAMS AND INPUTS //
	//////////////////////////////////////////////////////////////

	private SimpleMatrix computeTensorDeltaDown(SimpleMatrix deltaFull, SimpleMatrix leftVector, 
			SimpleMatrix rightVector, SimpleMatrix W,
			SimpleTensor Wt) {
		int subDim = leftVector.numRows();
		SimpleMatrix WTDelta = W.transpose().mult(deltaFull);
		SimpleMatrix WTDeltaNoBias = WTDelta.extractMatrix(0, (subDim * 2), 0, 1);
		SimpleMatrix leftDelta = new SimpleMatrix(subDim, 1);
		SimpleMatrix rightDelta = new SimpleMatrix(subDim, 1);
		double sliceGrad = 0.0;
		for (int slice=0; slice < subDim; slice++) {
			sliceGrad = deltaFull.get(slice);
			leftDelta = leftDelta.plus(Wt.getSlice(slice).mult(rightVector.scale(sliceGrad)));
			rightDelta = rightDelta.plus(Wt.getSlice(slice).transpose().mult(leftVector.scale(sliceGrad)));
		}
		return NeuralUtils.concatenate(leftDelta, rightDelta).plus(WTDeltaNoBias);
	}

	private SimpleTensor getTensorGradient(SimpleMatrix deltaFull, SimpleMatrix leftVector, 
			SimpleMatrix rightVector) {
		int subDim = deltaFull.numRows();
		double sliceGrad = 0.0;
		SimpleTensor Wt_df = new SimpleTensor(subDim, subDim, subDim);
		for (int slice=0; slice < subDim; slice++) {
			sliceGrad = deltaFull.get(slice);
			Wt_df.setSlice(slice, leftVector.scale(sliceGrad).mult(rightVector.transpose()));
		}
		return Wt_df;
	}

	/////////////////////////////////////////////////////////////////////////////
	// MULTI-SAMPLE FORWARD PROPAGATION THROUGH PHRASE TREE, TO COMPUTE        //
	// OUTPUT WITH RESPECT TO A UNIFORM DISTRIBUTION OVER HALF-MODELS.         //
	/////////////////////////////////////////////////////////////////////////////

	/**
	 * Multi-sample feedforward for approximating the dropout ensemble. This
	 * method samples multiple models and aggregrates the predictions made by
	 * each model, in order to compute a "pseudo-ensemble" prediction.
	 */
	public void multiPropagateTree(Tree tree, int samples) {
		double sampleScale = 1.0 / ((double) samples);
		initMultiPreds(tree);
		for (int i=0; i < samples; i++) {
			// This samples a droppy "child model" if model.useDropout is true
			// and uses the full "parent model" if model.useDropout is false.
			// For now, we will ignore the possible effects of weight-noise.
			setDropModel(tree, model.useDropout, false);
			evalPropTree(tree, sampleScale);
		}
		// Clean up lingering annotations/pointers to submodel stuff
		cleanTree(tree);
		return;
	}

	/**
	 * Run a sample feedforward through tree based on a sampled submodel.
	 */
	public void evalPropTree(Tree tree, double sampleScale) {
		// Get the annotations that determine how to feedforward through this node
		SimpleMatrix liveIdx = RNNCoreAnnotations.getLiveIdx(tree);
		SimpleMatrix binaryMatrix = RNNCoreAnnotations.getBinaryMatrix(tree);
		SimpleTensor binaryTensor = RNNCoreAnnotations.getBinaryTensor(tree);
		SimpleMatrix classyMatrix = RNNCoreAnnotations.getClassyMatrix(tree);
		// nodeVector will hold the "output" of this node (in latent space)
		SimpleMatrix nodeVector = null;
		SimpleMatrix nodePreVector = null;
		if (tree.isLeaf()) {
			throw new AssertionError("We should not have reached leaves in forwardPropagate");
		} else if (tree.isPreTerminal()) {
			// Get the word vector passed into this node by its dangling leaf
			String word = tree.children()[0].label().value();
			SimpleMatrix wordVector = model.getWordVector(word);
			// Slice the word vector to account for submodel sampling, and tanh it
			nodePreVector = sliceWordVector(liveIdx, wordVector);
		} else if (tree.children().length == 1) {
			// This might be important, so we'll just leave it here...
			throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
		} else if (tree.children().length == 2) {
			// Compute weights for each child, based on "words-in-subtree"
			double lWeight = 1.0;
			double rWeight = 1.0;
			if (model.op.wordWeight) {
				double lWC = (double) countWords(tree.children()[0]);
				double rWC = (double) countWords(tree.children()[1]);
				lWeight = lWC / (lWC + rWC);
				rWeight = rWC / (lWC + rWC);
			}
			// fprop child subtrees, to make their results available to this node.
			evalPropTree(tree.children()[0], sampleScale);
			evalPropTree(tree.children()[1], sampleScale);
			// Compute RNTN fprop through this node.
			SimpleMatrix lcVec = RNNCoreAnnotations.getNodeVector(tree.children()[0]).scale(lWeight);
			SimpleMatrix rcVec = RNNCoreAnnotations.getNodeVector(tree.children()[1]).scale(rWeight);
			SimpleMatrix jcVec = NeuralUtils.concatenateWithBias(lcVec, rcVec);
			if (model.op.useTensors) {
				SimpleMatrix tensorOut = binaryTensor.myBilinearProducts(lcVec, rcVec);
				nodePreVector = binaryMatrix.mult(jcVec).plus(tensorOut);
			} else {
				nodePreVector = binaryMatrix.mult(jcVec);
			}
		} else {
			throw new AssertionError("Tree not correctly binarized");
		}
		// Apply activation function to nodePreVector to get nodeVector
		nodeVector = fpropActivation(nodePreVector);
		// Compute the prediction given by the current submodel
		SimpleMatrix currentPreds = classyMatrix.mult(NeuralUtils.concatenateWithBias(nodeVector));
		if (model.useBinary) {
			// When running in binary mode, we only want to use the value in
			// linearPreds[2]. But, we'll move elsewhere, to let the normal
			// prediction code ignore the binary/fine-grained distinction.
			currentPreds.set(0, -currentPreds.get(2));
			currentPreds.set(4, currentPreds.get(2));
			currentPreds.set(1, 0.0);
			currentPreds.set(2, 0.0);
			currentPreds.set(3, 0.0);
		}
		// Get the current prediction accumulator for this node, and update it
		SimpleMatrix multiPreds = RNNCoreAnnotations.getMultiPreds(tree);
		multiPreds.set(multiPreds.plus(currentPreds.scale(sampleScale)));
		// Compute classifier outputs/predictions for this node.
		SimpleMatrix predictions = multiPreds.scale(1.0);
		// Store the values needed to backprop through this node.
		int index = getPredictedClass(predictions);
		int binPred = getBinaryPrediction(predictions);
		if (!(tree.label() instanceof CoreLabel)) {
			throw new AssertionError("Expected CoreLabels in the nodes");
		}
		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.Predictions.class, predictions);
		label.set(RNNCoreAnnotations.PredictedClass.class, index);
		label.set(RNNCoreAnnotations.BinaryPrediction.class, binPred);
		label.set(RNNCoreAnnotations.NodeVector.class, nodeVector);
		label.set(RNNCoreAnnotations.NodePreVector.class, nodePreVector);
	}

	////////////////////////////////////////////
	// NOISE-PERTURBATION AUXILIARY FUNCTIONS //
	////////////////////////////////////////////

	/**
	 * Compute a noise-perturbed submodel to use for this tree. Note that all
	 * tree nodes for a phrase will use the same noise-perturbed RNTN, which
	 * corresponds to drawing models uniformly at random from the collection of
	 * all RNTN models defined over subspaces of the full embedding space, and
	 * then (maybe) perturbing them with some Gaussian noise.
	 */
	public void setDropModel(Tree tree, boolean doDrop, boolean addNoise) {
		// This is used at the root node, so here we make this tree's mask.
		SimpleMatrix liveIdx = setLiveIdx(doDrop);
		// Use the sampled dropout mask to extract subtensors for faster fprop
		// and backprop. 
		SimpleMatrix binaryMatrix = sliceBinaryMatrix(liveIdx, model.binaryMatrix);
		SimpleTensor binaryTensor = sliceBinaryTensor(liveIdx, model.binaryTensor);
		SimpleMatrix classyMatrix = sliceClassyMatrix(liveIdx, model.classyMatrix);
		// For testing... add some noise to the parameter matrices...
		if (addNoise == true) {
			double noiseStd = model.op.trainOptions.regParamNoise;
			if (noiseStd > 1e-4) {
				SimpleMatrix randBM = NeuralUtils.randomGaussian(binaryMatrix.numRows(), 
						binaryMatrix.numCols(), model.rand);
				SimpleTensor randBT = SimpleTensor.randomGaussian(binaryTensor.numRows(),
						binaryTensor.numCols(), binaryTensor.numSlices(), model.rand);
				SimpleMatrix randCM = NeuralUtils.randomGaussian(classyMatrix.numRows(),
						classyMatrix.numCols(), model.rand);
				binaryMatrix.set(binaryMatrix.plus(randBM.scale(noiseStd)));
				binaryTensor.set(binaryTensor.plus(randBT.scale(noiseStd)));
				classyMatrix.set(classyMatrix.plus(randCM.scale(noiseStd)));
			}
		}

		// Recursively set node properties in this phrase tree.
		if (!(tree.label() instanceof CoreLabel)) {
			throw new AssertionError("Expected CoreLabels in the nodes");
		}
		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.BinaryMatrix.class, binaryMatrix);
		label.set(RNNCoreAnnotations.BinaryTensor.class, binaryTensor);
		label.set(RNNCoreAnnotations.ClassyMatrix.class, classyMatrix);
		label.set(RNNCoreAnnotations.LiveIdx.class, liveIdx);
		if (tree.children().length == 2) {
			setDropModel(tree.children()[0], binaryMatrix, binaryTensor, classyMatrix, liveIdx);
			setDropModel(tree.children()[1], binaryMatrix, binaryTensor, classyMatrix, liveIdx);
		} else if (tree.isPreTerminal()) {
			setDropModel(tree.children()[0], binaryMatrix, binaryTensor, classyMatrix, liveIdx);
		}
		return;
	}

	public void setDropModel(Tree tree, SimpleMatrix binaryMatrix, SimpleTensor binaryTensor, 
			SimpleMatrix classyMatrix, SimpleMatrix liveIdx) {
		// Set the properties for this node, then recurse
		if (!(tree.label() instanceof CoreLabel)) {
			throw new AssertionError("Expected CoreLabels in the nodes");
		}
		CoreLabel label = (CoreLabel) tree.label();
		label.set(RNNCoreAnnotations.BinaryMatrix.class, binaryMatrix);
		label.set(RNNCoreAnnotations.BinaryTensor.class, binaryTensor);
		label.set(RNNCoreAnnotations.ClassyMatrix.class, classyMatrix);
		label.set(RNNCoreAnnotations.LiveIdx.class, liveIdx);
		if (tree.children().length == 2) {
			setDropModel(tree.children()[0], binaryMatrix, binaryTensor, classyMatrix, liveIdx);
			setDropModel(tree.children()[1], binaryMatrix, binaryTensor, classyMatrix, liveIdx);
		} else if (tree.isPreTerminal()) {
			setDropModel(tree.children()[0], binaryMatrix, binaryTensor, classyMatrix, liveIdx);
		}
		return;
	}

	/**
	 * This method selects a random subset of (numHid / 2) dimensions to
	 * retain for random submodel sampling. I.e. this does dropout on the
	 * latent dimensions of the model, constrained to always drop exactly half
	 * of the available latent dimensions.
	 */ 
	public SimpleMatrix setLiveIdx(boolean doDrop){
		int keepCount = model.numHid;
		if (doDrop) {
			keepCount = keepCount / 2;
		}
		// Generate a random binary "half-mask"
		SimpleMatrix maskMat = new SimpleMatrix(model.numHid, 1);
		Collections.shuffle(dropIdx);
		for (int i=0; i < keepCount; i++) {
			maskMat.set(dropIdx.get(i), 0, 1.0);
		}
		// Convert the binary masking matrix into a live index list
		SimpleMatrix liveIdx = new SimpleMatrix(keepCount, 1);
		int j = 0;
		for (int i=0; i < model.numHid; i++) {
			if (maskMat.get(i) > 0.1) {
				liveIdx.set(j, 0, i);
				j += 1;
			}
		}
		return liveIdx;
	}

	////////////////////////////////////////////////
	// BASIC TREE ANNOTATION MANAGEMENT FUNCTIONS //
	////////////////////////////////////////////////

	/**
	 * Prepare tree for computation of multi-sample predictions.
	 */
	public void initMultiPreds(Tree tree) {
		if (!(tree.label() instanceof CoreLabel)) {
			throw new AssertionError("Expected CoreLabels in the nodes");
		}
		CoreLabel label = (CoreLabel) tree.label();
		// Create a vector to hold multi-sample mean predictions
		SimpleMatrix multiPreds = new SimpleMatrix(model.numClasses, 1);
		label.set(RNNCoreAnnotations.MultiPreds.class, multiPreds);
		if (tree.children().length == 2) {
			initMultiPreds(tree.children()[0]);
			initMultiPreds(tree.children()[1]);
		} else if (tree.isPreTerminal()) {
			initMultiPreds(tree.children()[0]);
		}
		return;
	}

	/**
	 * Destroy dangling annotations on the tree.
	 */
	public void cleanTree(Tree tree) {
		if (!(tree.label() instanceof CoreLabel)) {
			throw new AssertionError("Expected CoreLabels in the nodes");
		}
		CoreLabel label = (CoreLabel) tree.label();
		// Clear annotations from tree that are not needed for scoring.
		SimpleMatrix dummyMatrix = null;
		SimpleTensor dummyTensor = null;
		label.set(RNNCoreAnnotations.BinaryMatrix.class, dummyMatrix);
		label.set(RNNCoreAnnotations.BinaryTensor.class, dummyTensor);
		label.set(RNNCoreAnnotations.ClassyMatrix.class, dummyMatrix);
		label.set(RNNCoreAnnotations.LiveIdx.class, dummyMatrix);
		if (tree.children().length == 2) {
			cleanTree(tree.children()[0]);
			cleanTree(tree.children()[1]);
		} else if (tree.isPreTerminal()) {
			cleanTree(tree.children()[0]);
		}
		return;
	}

	////////////////////////////////////////////////////
	// MATRIX/TENSOR SLICING AND UN-SLICING FUNCTIONS //
	////////////////////////////////////////////////////

	/**
	 * Extract entries from a word vector based on liveIdx.
	 */
	public SimpleMatrix sliceWordVector(SimpleMatrix liveIdx, SimpleMatrix wordVector) {
		int bigSize = model.numHid;
		int smallSize = liveIdx.numRows();
		int bigRI = 0;
		if (bigSize == smallSize) {
			return wordVector;
		}
		SimpleMatrix subVector = new SimpleMatrix(smallSize, 1);
		for (int smallRI=0; smallRI < smallSize; smallRI++) {
			bigRI = (int) liveIdx.get(smallRI);
			subVector.set(smallRI, wordVector.get(bigRI));
		}
		return subVector;
	}

	/**
	 * Dextract entries from a word vector based on liveIdx.
	 */
	public SimpleMatrix unsliceWordVector(SimpleMatrix liveIdx, SimpleMatrix subVector) {
		int bigSize = model.numHid;
		int smallSize = liveIdx.numRows();
		int bigRI = 0;
		if (bigSize == smallSize) {
			return subVector;
		}
		SimpleMatrix wordVector = new SimpleMatrix(bigSize, 1);
		for (int smallRI=0; smallRI < smallSize; smallRI++) {
			bigRI = (int) liveIdx.get(smallRI);
			wordVector.set(bigRI, subVector.get(smallRI));
		}
		return wordVector;
	}

	/**
	 * Extract the binaryMatrix required for computing feedforward and backprop
	 * through an RNTN phrase tree using the submodel determined by liveIdx. The
	 * slicing structure assumes that the left and right vectors will be stacked
	 * vertically, with a bias appended below them, and the left/right/output
	 * vectors will all be sliced with the same liveIdx.
	 */
	public SimpleMatrix sliceBinaryMatrix(SimpleMatrix liveIdx, SimpleMatrix binaryMatrix) {
		int bigRows = model.numHid;
		int bigCols = (2 * model.numHid) + 1;
		int lBigSize = bigRows;
		int rBigSize = bigRows;
		int smallRows = liveIdx.numRows();
		int lSmallSize = liveIdx.numRows();
		int rSmallSize = liveIdx.numRows();
		int smallCols = lSmallSize + rSmallSize + 1;
		int bigRI = 0;
		int bigCI = 0;
		if (bigCols == smallCols) {
			return binaryMatrix;
		}
		// Now, extract the appropriate submatrix from binaryMatrix
		SimpleMatrix subMatrix = new SimpleMatrix(smallRows, smallCols);
		for (int smallRI=0; smallRI < smallRows; smallRI++) {
			bigRI = (int) liveIdx.get(smallRI);
			for (int smallCI=0; smallCI < smallCols; smallCI++) {
				if (smallCI < lSmallSize) {
					// We're in the leftVector part of submatrix...
					bigCI = (int) liveIdx.get(smallCI);
				} else if (smallCI < (lSmallSize + rSmallSize)) {
					// We're in the rightVector part of submatrix...
					bigCI = ((int) liveIdx.get(smallCI - lSmallSize)) + lBigSize;
				} else {
					// We're in the bias part of submatrix....
					bigCI = bigCols - 1;
				}
				subMatrix.set(smallRI, smallCI, binaryMatrix.get(bigRI, bigCI));
			}
		}
		return subMatrix;
	}

	/**
	 * Dextract the binaryMatrix required for computing feedforward and backprop
	 * through an RNTN phrase tree using the submodel determined by liveIdx. The
	 * slicing structure assumes that the left and right vectors will be stacked
	 * vertically, with a bias appended below them, and the left/right/output
	 * vectors will all be sliced with the same liveIdx.
	 */
	public SimpleMatrix unsliceBinaryMatrix(SimpleMatrix liveIdx, SimpleMatrix subMatrix) {
		int bigRows = model.numHid;
		int bigCols = (2 * model.numHid) + 1;
		int lBigSize = bigRows;
		int rBigSize = bigRows;
		int smallRows = liveIdx.numRows();
		int lSmallSize = liveIdx.numRows();
		int rSmallSize = liveIdx.numRows();
		int smallCols = lSmallSize + rSmallSize + 1;
		int bigRI = 0;
		int bigCI = 0;
		if (bigCols == smallCols) {
			return subMatrix;
		}
		// Now, dextract the appropriate submatrix from subBinaryMatrix
		SimpleMatrix binaryMatrix = new SimpleMatrix(bigRows, bigCols);
		for (int smallRI=0; smallRI < smallRows; smallRI++) {
			bigRI = (int) liveIdx.get(smallRI);
			for (int smallCI=0; smallCI < smallCols; smallCI++) {
				if (smallCI < lSmallSize) {
					// We're in the leftVector part of submatrix...
					bigCI = (int) liveIdx.get(smallCI);
				} else if (smallCI < (lSmallSize + rSmallSize)) {
					// We're in the rightVector part of submatrix...
					bigCI = ((int) liveIdx.get(smallCI - lSmallSize)) + lBigSize;
				} else {
					// We're in the bias part of submatrix....
					bigCI = bigCols - 1;
				}
				binaryMatrix.set(bigRI, bigCI, subMatrix.get(smallRI, smallCI));
			}
		}
		return binaryMatrix;
	}

	/**
	 * Extract the binaryTensor required for computing feedforward and backprop
	 * through an RNTN phrase tree using the submodel determined by liveIdx. The
	 * slicing structure assumes the left vector, right vector, and group output
	 * will all be sliced with the same liveIdx.
	 */
	public SimpleTensor sliceBinaryTensor(SimpleMatrix liveIdx, SimpleTensor binaryTensor) {
		int bigSize = model.numHid; // full binaryTensor is cubic (i.e. NxNxN)
		int smallSize = liveIdx.numRows(); // subTensor will be a smaller cube
		int bigRI = 0;
		int bigCI = 0;
		int bigSI = 0;
		if (bigSize == smallSize) {
			return binaryTensor;
		}
		// Now, extract the appropriate subtensor from binaryTensor
		SimpleTensor subTensor = new SimpleTensor(smallSize, smallSize, smallSize);
		for (int smallRI=0; smallRI < smallSize; smallRI++) {
			bigRI = (int) liveIdx.get(smallRI);
			for (int smallCI=0; smallCI < smallSize; smallCI++) {
				bigCI = (int) liveIdx.get(smallCI);
				for (int smallSI=0; smallSI < smallSize; smallSI++) {
					bigSI = (int) liveIdx.get(smallSI);
					subTensor.set(smallRI, smallCI, smallSI, binaryTensor.get(bigRI, bigCI, bigSI));
				}
			}
		}
		return subTensor;
	}

	/**
	 * Dextract the binaryTensor required for computing feedforward and backprop
	 * through an RNTN phrase tree using the submodel determined by liveIdx. The
	 * slicing structure assumes the left vector, right vector, and group output
	 * will all be sliced with the same liveIdx.
	 */
	public SimpleTensor unsliceBinaryTensor(SimpleMatrix liveIdx, SimpleTensor subTensor) {
		int bigSize = model.numHid; // full binaryTensor is cubic (i.e. NxNxN)
		int smallSize = liveIdx.numRows(); // subTensor will be a smaller cube
		int bigRI = 0;
		int bigCI = 0;
		int bigSI = 0;
		if (bigSize == smallSize) {
			return subTensor;
		}
		// Now, extract the appropriate subtensor from binaryTensor
		SimpleTensor binaryTensor = new SimpleTensor(bigSize, bigSize, bigSize);
		for (int smallRI=0; smallRI < smallSize; smallRI++) {
			bigRI = (int) liveIdx.get(smallRI);
			for (int smallCI=0; smallCI < smallSize; smallCI++) {
				bigCI = (int) liveIdx.get(smallCI);
				for (int smallSI=0; smallSI < smallSize; smallSI++) {
					bigSI = (int) liveIdx.get(smallSI);
					binaryTensor.set(bigRI, bigCI, bigSI, subTensor.get(smallRI, smallCI, smallSI));
				}
			}
		}
		return binaryTensor;
	}

	/**
	 * Extract the classyMatrix required for computing feedforward and backprop
	 * through an RNTN phrase tree using the submodel determined by liveIdx. The
	 * slicing structure assumes a bias will be appended to the end of each
	 * nodeVector prior to classification.
	 */
	public SimpleMatrix sliceClassyMatrix(SimpleMatrix liveIdx, SimpleMatrix classyMatrix) {
		int bigRows = model.numClasses;
		int bigCols = model.numHid + 1;
		int smallRows = model.numClasses;
		int smallCols = liveIdx.numRows() + 1;
		int bigRI = 0;
		int bigCI = 0;
		if (bigCols == smallCols) {
			return classyMatrix;
		}
		// Now, extract the appropriate submatrix from classyMatrix
		SimpleMatrix subMatrix = new SimpleMatrix(smallRows, smallCols);
		for (int smallRI=0; smallRI < smallRows; smallRI++) {
			bigRI = smallRI;
			for (int smallCI=0; smallCI < smallCols; smallCI++) {
				if (smallCI < (smallCols - 1)) {
					// We're in the non-bias part of submatrix...
					bigCI = (int) liveIdx.get(smallCI);
				} else {
					// We're in the bias part of submatrix....
					bigCI = bigCols - 1;
				}
				subMatrix.set(smallRI, smallCI, classyMatrix.get(bigRI, bigCI));
			}
		}
		return subMatrix;
	}

	/**
	 * Dextract the classyMatrix required for computing feedforward and backprop
	 * through an RNTN phrase tree using the submodel determined by liveIdx. The
	 * slicing structure assumes a bias will be appended to the end of each
	 * nodeVector prior to classification.
	 */
	public SimpleMatrix unsliceClassyMatrix(SimpleMatrix liveIdx, SimpleMatrix subMatrix) {
		int bigRows = model.numClasses;
		int bigCols = model.numHid + 1;
		int smallRows = model.numClasses;
		int smallCols = liveIdx.numRows() + 1;
		int bigRI = 0;
		int bigCI = 0;
		if (bigCols == smallCols) {
			return subMatrix;
		}
		// Now, extract the appropriate submatrix from classyMatrix
		SimpleMatrix classyMatrix = new SimpleMatrix(bigRows, bigCols);
		for (int smallRI=0; smallRI < smallRows; smallRI++) {
			bigRI = smallRI;
			for (int smallCI=0; smallCI < smallCols; smallCI++) {
				if (smallCI < (smallCols - 1)) {
					// We're in the non-bias part of submatrix...
					bigCI = (int) liveIdx.get(smallCI);
				} else {
					// We're in the bias part of submatrix....
					bigCI = bigCols - 1;
				}
				classyMatrix.set(bigRI, bigCI, subMatrix.get(smallRI, smallCI));
			}
		}
		return classyMatrix;
	}

	/**
	 * Do simple kind of parallel feedforward and backprop through a batch of
	 * trees. This modifies the objects which are passed in, and returns the
	 * sum of the errors for the trees in the given minibatch.
	 */
	public double ffbpTrees(List<Tree> batchTrees, final SimpleMatrix binaryMD, final SimpleTensor binaryTD, 
			final SimpleMatrix classyMD, final Map<String, SimpleMatrix> wordVD)
					throws InterruptedException, ExecutionException {

		// Stuff for managing multithreading...
		int threads = Runtime.getRuntime().availableProcessors();
		ExecutorService service = Executors.newFixedThreadPool(threads);
		List<Future<Double>> futures =  new ArrayList<Future<Double>>();

		// Run feedforward on each tree in the minibatch.
		for (final Tree t : batchTrees) {
			Callable<Double> callable = new Callable<Double>() {
				public Double call() throws Exception {
					// process your input here and compute the output
					setDropModel(t, model.useDropout, true);
					forwardPropagateTree(t, true);
					Double dummyDouble = new Double(0.0);
					return dummyDouble;
				}
			};
			futures.add(service.submit(callable));
		}
		// Synchronize to make sure feedforward for all trees is done.
		for (Future<Double> future : futures) {
			future.get();
		}

		// Now, do backprop...
		for (final Tree t : batchTrees) {
			Callable<Double> callable = new Callable<Double>() {
				public Double call() throws Exception {
					// process your input here and compute the output
					backpropDerivativesAndError(t, binaryMD, binaryTD, classyMD, wordVD);
					Double treeError = new Double(sumError(t));
					return treeError;
				}
			};
			futures.add(service.submit(callable));
		}
		double error = 0.0;
		for (Future<Double> future : futures) {
			// Synchronize to make sure backprop in all trees is done
			error = error + future.get();
		}

		// shutdown worker/thread pool
		service.shutdown();
		return error;
	}

}
