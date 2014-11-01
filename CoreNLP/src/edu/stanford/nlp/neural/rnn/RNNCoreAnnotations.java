package edu.stanford.nlp.neural.rnn;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.neural.SimpleTensor;

public class RNNCoreAnnotations {

  private RNNCoreAnnotations() {} // only static members

  ////////////////////////////////////////////////////////
  // GETTERS/SETTERS FOR TREE-SPECIFIC MODEL PARAMETERS //
  ////////////////////////////////////////////////////////

  /**
   * Used to denote binaryMatrix of the submodel used at a particular node.
   */
  public static class BinaryMatrix implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }

  public static SimpleMatrix getBinaryMatrix(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached binaryMatrix");
    }
    return ((CoreLabel) label).get(BinaryMatrix.class);
  }
  
  /**
   * Used to denote binaryTensor of the submodel used at a particular node.
   */
  public static class BinaryTensor implements CoreAnnotation<SimpleTensor> {
    public Class<SimpleTensor> getType() {
      return SimpleTensor.class;
    }
  }

  public static SimpleTensor getBinaryTensor(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached binaryTensor");
    }
    return ((CoreLabel) label).get(BinaryTensor.class);
  }

  /**
   * Used to denote classyMatrix of the submodel used at a particular node.
   */
  public static class ClassyMatrix implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }

  public static SimpleMatrix getClassyMatrix(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached classyMatrix");
    }
    return ((CoreLabel) label).get(ClassyMatrix.class);
  }

  /////////////////////////////////////////////////////////////
  // GETTERS/SETTERS FOR VECTORS PRODUCED DURING FEEDFORWARD //
  /////////////////////////////////////////////////////////////

  /**
   * Used to denote the vector at a particular node
   */
  public static class NodeVector implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }
  
  public static SimpleMatrix getNodeVector(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached node vector");
    }
    return ((CoreLabel) label).get(NodeVector.class);
  }

  /**
   * Used to denote the pre-activation vector at a particular node
   */
  public static class NodePreVector implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }
  
  public static SimpleMatrix getNodePreVector(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached node pre-vector");
    }
    return ((CoreLabel) label).get(NodePreVector.class);
  }

  /**
   * Used to denote the noisy version of nodeVector.
   */
  public static class NoisyNodeVector implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }
  
  public static SimpleMatrix getNoisyNodeVector(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached noisyNodeVector");
    }
    return ((CoreLabel) label).get(NoisyNodeVector.class);
  }

  /**
   * A list of non-zero indices in this node's drop mask.
   */
  public static class LiveIdx implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }
  
  public static SimpleMatrix getLiveIdx(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached liveIdx");
    }
    return ((CoreLabel) label).get(LiveIdx.class);
  }

  /////////////////////////////////////////////////////////////////
  // GETTERS/SETTERS FOR TRACKING CLASS PREDICTIONS AT EACH NODE //
  /////////////////////////////////////////////////////////////////

  /**
   * Used to denote this node's multi-sample predictions.
   */
  public static class MultiPreds implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }
  
  public static SimpleMatrix getMultiPreds(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached multiPreds");
    }
    return ((CoreLabel) label).get(MultiPreds.class);
  }

  /**
   * Used to denote a vector of predictions at a particular node
   */
  public static class Predictions implements CoreAnnotation<SimpleMatrix> {
    public Class<SimpleMatrix> getType() {
      return SimpleMatrix.class;
    }
  }
  
  public static SimpleMatrix getPredictions(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached predictions");
    }
    return ((CoreLabel) label).get(Predictions.class);
  }

  /**
   * argmax of the Predictions
   */
  public static class PredictedClass implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {
      return Integer.class;
    }
  }

  public static int getPredictedClass(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached gold class");
    }
    return ((CoreLabel) label).get(PredictedClass.class);
  }

  /**
   * Binary predictions, based on classes 0/1 being "negative" and classes
   * 3/4 being "positive". This assumes five possible classes 0-4, with 2
   * indicating "neutral" sentiment. The "negative" class is indicated by 0
   * the "positive" class is indicated by 1. When "binarizing" the predictions
   * for each observation, output for class 2 (i.e. neutral) is ignored.
   */
  public static class BinaryPrediction implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {
      return Integer.class;
    }
  }

  public static int getBinaryPrediction(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached binary prediction");
    }
    return ((CoreLabel) label).get(BinaryPrediction.class);
  }

  /**
   * The index of the correct class
   */
  public static class GoldClass implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {
      return Integer.class;
    }
  }

  public static int getGoldClass(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached gold class");
    }
    return ((CoreLabel) label).get(GoldClass.class);
  }

  public static void setGoldClass(Tree tree, int goldClass) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached gold class");
    }
    ((CoreLabel) label).set(GoldClass.class, goldClass);
  }

  ////////////////////////////////////////////////////////////////////////
  // GETTERS/SETTERS FOR TRACKING DIFFERENT TYPES OF ERROR AT EACH NODE //
  ////////////////////////////////////////////////////////////////////////

  public static class PredictionError implements CoreAnnotation<Double> {
    public Class<Double> getType() {
      return Double.class;
    }
  }

  public static double getPredictionError(Tree tree) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to get the attached prediction error");
    }
    return ((CoreLabel) label).get(PredictionError.class);
  }

  public static void setPredictionError(Tree tree, double error) {
    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("CoreLabels required to set the attached prediction error");
    }
    ((CoreLabel) label).set(PredictionError.class, error);
  }

}
