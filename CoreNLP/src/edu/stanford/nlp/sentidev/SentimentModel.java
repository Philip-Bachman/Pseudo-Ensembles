package edu.stanford.nlp.sentidev;

import java.io.Serializable;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.neural.Embedding;
import edu.stanford.nlp.neural.NeuralUtils;
import edu.stanford.nlp.neural.SimpleTensor;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.TwoDimensionalMap;
import edu.stanford.nlp.util.TwoDimensionalSet;

public class SentimentModel implements Serializable {
	/**
	 * Nx2N+1, where N is the size of the word vectors
	 */
	public SimpleMatrix binaryMatrix;

	/**
	 * 2Nx2NxN, where N is the size of the word vectors
	 */
	public SimpleTensor binaryTensor;

	/**
	 * CxN+1, where N = size of word vectors, C is the number of classes
	 */
	public SimpleMatrix classyMatrix;

	/**
	 * V x N, where N = size of word vectors and V = size of vocabulary
	 */
	public Map<String, SimpleMatrix> wordVectors;

	/**
	 * How many classes the RNN is built to test against
	 */
	public final int numClasses;

	/**
	 * Dimension of hidden layers, size of word vectors, etc
	 */
	public final int numHid;
	public final int subDim;

	/** How many elements a transformation matrix has */
	public final int binaryMatrixSize;
	/** How many elements the binary transformation tensors have */
	public final int binaryTensorSize;
	/** How many elements a classification matrix has */
	public final int classyMatrixSize;

	/**
	 * we just keep this here for convenience
	 */
	transient SimpleMatrix identity;

	/** 
	 * A random number generator - keeping it here lets us reproduce results
	 */
	final Random rand;

	public static final String UNKNOWN_WORD = "*UNK*";

	/**
	 * Will store various options specific to this model
	 */
	final RNNOptions op;

	/**
	 * useDropout tells if this model will train/test with submodel sampling.
	 */
	public final boolean useDropout;

	/**
	 * useBinary tells whether to use binary or fine-grained loss.
	 */
	public final boolean useBinary;


	/**
	 * The traditional way of initializing an empty model suitable for training.
	 */
	public SentimentModel(RNNOptions op, List<Tree> trainingTrees) {
		// Set various attributes of this model
		this.op = op;
		this.useDropout = op.useDropout;
		this.useBinary = op.useBinary;
		this.numClasses = op.numClasses;

		// Attach a source of randomness to this model
		rand = new Random(op.randomSeed);

		// Initialize the word embedding vectors for this model
		if (op.randomWordVectors) {
			initRandomWordVectors(trainingTrees);
		} else {
			readWordVectors();
		}
		if (op.numHid > 0) {
			// Set numHid based on options structure
			this.numHid = op.numHid;
		} else {
			//  Set numHid based on preexisting word vector dimension
			int size = 0;
			for (SimpleMatrix vector : wordVectors.values()) {
				size = vector.getNumElements();
				break;
			}
			this.numHid = size;
		}

		this.identity = SimpleMatrix.identity(numHid);
		if (this.useDropout) {
			this.subDim = numHid / 2;
		} else {
			this.subDim = numHid;
		}

		this.binaryMatrix = randomTransformMatrix();
		this.binaryTensor = randomBinaryTensor();
		this.classyMatrix = randomClassificationMatrix();

		// Record information about transform parameters
		this.binaryMatrixSize = numHid * (2 * numHid + 1);
		this.classyMatrixSize = this.numClasses * (numHid + 1);
		if (op.useTensors) {
			this.binaryTensorSize = numHid * numHid * numHid;
		} else {
			this.binaryTensorSize = numHid * numHid * numHid;
			this.binaryTensor = new SimpleTensor(numHid, numHid, numHid);
		}

	}

	////////////////////////////////////////////////////
	// INITIALIZE TRANSFORM AND CLASSIFIER PARAMETERS //
	////////////////////////////////////////////////////

	SimpleTensor randomBinaryTensor() {
		double range = 1.0 / (2.0 * numHid);
		SimpleTensor tensor = SimpleTensor.random(numHid, numHid, numHid, -range, range, rand);
		return tensor.scale(op.trainOptions.scalingForInit);
	}

	SimpleMatrix randomTransformMatrix() {
		SimpleMatrix binary = new SimpleMatrix(numHid, numHid * 2 + 1);
		// bias column values are initialized zero
		binary.insertIntoThis(0, 0, randomTransformBlock());
		binary.insertIntoThis(0, numHid, randomTransformBlock());
		return binary.scale(op.trainOptions.scalingForInit);
	}

	SimpleMatrix randomTransformBlock() {
		double range = 1.0 / (Math.sqrt((double) numHid) * 2.0);
		return SimpleMatrix.random(numHid,numHid,-range,range,rand).plus(identity.scale(1.0));
	}

	SimpleMatrix randomClassificationMatrix() {
		SimpleMatrix score = new SimpleMatrix(numClasses, numHid + 1);
		// Leave the bias column with 0 values
		double range = 1.0 / (Math.sqrt((double) numHid));
		score.insertIntoThis(0, 0, SimpleMatrix.random(numClasses, numHid, -range, range, rand));
		return score.scale(op.trainOptions.scalingForInit);
	}

	///////////////////////////////////////
	// INITIALIZE WORD EMBEDDING VECTORS //
	///////////////////////////////////////

	SimpleMatrix randomWordVector() {
		return randomWordVector(op.numHid, rand);
	}

	static SimpleMatrix randomWordVector(int size, Random rand) {
		SimpleMatrix wordVector = NeuralUtils.randomGaussian(size, 1, rand);
		return wordVector;
	}

	void initRandomWordVectors(List<Tree> trainingTrees) {
		if (op.numHid == 0) {
			throw new RuntimeException("Cannot create random word vectors for an unknown numHid");
		}
		double wfThresh = (double)op.minWordFreq - 0.1;
		Set<String> words = Generics.newHashSet();
		Counter<String> wordCounter = new ClassicCounter<String>();
		// set the unknown word token to have an adequate count
		words.add(UNKNOWN_WORD);
		wordCounter.setCount(UNKNOWN_WORD, (double)op.minWordFreq);
		// count the occurrences of each unique token in the training trees, with
		// the words forced to all lower-case if op.lowercaseWordVectors is true.
		for (Tree tree : trainingTrees) {
			List<Tree> leaves = tree.getLeaves();
			for (Tree leaf : leaves) {
				String word = leaf.label().value();
				if (op.lowercaseWordVectors) {
					word = word.toLowerCase();
				}
				words.add(word);
				wordCounter.incrementCount(word);
			}
		}
		// generate the final set of words with sufficient frequency
		this.wordVectors = Generics.newTreeMap();
		int vocabWords = 0;
		int numDiscards = 0;
		for (String word : words) {
			SimpleMatrix vector = randomWordVector();
			if (wordCounter.getCount(word) > wfThresh) {
				wordVectors.put(word, vector);
				vocabWords += 1;
			} else {
				numDiscards += 1;
			}
		}
		System.out.println("KEPT WORDS: " + vocabWords + ", DISCARDED WORDS: " + numDiscards);
	}

	void readWordVectors() {
		Embedding embedding = new Embedding(op.wordVectors, op.numHid);
		this.wordVectors = Generics.newTreeMap();
		for (String word : embedding.keySet()) {
			// TODO: factor out unknown word vector code from DVParser
			wordVectors.put(word, embedding.get(word));
		}

		String unkWord = op.unkWord;
		SimpleMatrix unknownWordVector = wordVectors.get(unkWord);
		wordVectors.put(UNKNOWN_WORD, unknownWordVector);
		if (unknownWordVector == null) {
			throw new RuntimeException("Unknown word vector not specified in the word vector file");
		}

	}

	///////////////////////////////////////////////////////////
	// ASSORTED STUFF THAT MAY OR MAY NOT DO STUFF TO THINGS //
	///////////////////////////////////////////////////////////

	public int totalParamSize() {
		int totalSize = 0;
		// binaryTensorSize was set to 0 if useTensors=false
		totalSize = binaryMatrixSize + binaryTensorSize + classyMatrixSize;
		totalSize += wordVectors.size() * numHid;
		return totalSize;
	}

	public double[] paramsToVector() {
		int totalSize = totalParamSize();
		List<SimpleMatrix> paramIterator = Generics.newArrayList();
		paramIterator.add(binaryMatrix);
		paramIterator.add(classyMatrix);
		return NeuralUtils.paramsToVector(totalSize, paramIterator.iterator(),
				binaryTensor.iteratorSimpleMatrix(), wordVectors.values().iterator());
	}

	public void vectorToParams(double[] theta) {
		List<SimpleMatrix> paramIterator = Generics.newArrayList();
		paramIterator.add(binaryMatrix);
		paramIterator.add(classyMatrix);
		NeuralUtils.vectorToParams(theta, paramIterator.iterator(),
				binaryTensor.iteratorSimpleMatrix(), wordVectors.values().iterator());
	}

	public SimpleMatrix getWordVector(String word) {
		return wordVectors.get(getVocabWord(word));
	}

	public String getVocabWord(String word) {
		if (op.lowercaseWordVectors) {
			word = word.toLowerCase();
		}
		if (wordVectors.containsKey(word)) {
			return word;
		}
		// TODO: go through unknown words here
		return UNKNOWN_WORD;
	}

	public String basicCategory(String category) {
		return "";
	}

	public SimpleMatrix getUnaryClassification(String category) {
		return classyMatrix; 
	}

	public SimpleMatrix getBinaryClassification(String left, String right) {
		return classyMatrix;
	}

	public SimpleMatrix getBinaryTransform(String left, String right) {
		return binaryMatrix;
	}

	public SimpleTensor getBinaryTensor(String left, String right) {
		return binaryTensor;
	}

	public void saveSerialized(String path) {
		try {
			IOUtils.writeObjectToFile(this, path);
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		}
	}

	public static SentimentModel loadSerialized(String path) {
		try {
			return IOUtils.readObjectFromURLOrClasspathOrFileSystem(path);
		} catch (IOException e) {
			throw new RuntimeIOException(e);
		} catch (ClassNotFoundException e) {
			throw new RuntimeIOException(e);
		}
	}

	private static final long serialVersionUID = 1;
}
