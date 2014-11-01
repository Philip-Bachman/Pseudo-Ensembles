package edu.stanford.nlp.neural;

import java.io.File;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.Filter;

/**
 * Includes a bunch of utility methods usable by projects which use
 * RNN, such as the parser and sentiment models.  Some methods convert
 * iterators of SimpleMatrix objects to and from a vector.  Others are
 * general utility methods on SimpleMatrix objects.
 *
 * @author John Bauer
 * @author Richard Socher
 * @author Thang Luong
 */
public class NeuralUtils {
  private NeuralUtils() {} // static methods only

  private static double[] funkySlope = {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125};
  private static double[] funkyStart = {0.0, 1.0, 1.50, 1.750, 1.8750, 1.93750, 1.968750, 1.9843750};

  /**
   * Convert a file into a text matrix.  The expected format one row
   * per line, one entry per column.  Not too efficient for large
   * matrices, but you shouldn't store large matrices in text files
   * anyway.  This specific format is not supported by ejml, which
   * expects the number of rows and columns in its text matrices.
   */
  public static SimpleMatrix loadTextMatrix(String path) {
    return convertTextMatrix(IOUtils.slurpFileNoExceptions(path));
  }

  /**
   * Convert a file into a text matrix.  The expected format one row
   * per line, one entry per column.  Not too efficient for large
   * matrices, but you shouldn't store large matrices in text files
   * anyway.  This specific format is not supported by ejml, which
   * expects the number of rows and columns in its text matrices.
   */
  public static SimpleMatrix loadTextMatrix(File file) {
    return convertTextMatrix(IOUtils.slurpFileNoExceptions(file));
  }

  public static SimpleMatrix convertTextMatrix(String text) {
    List<String> lines = CollectionUtils.filterAsList(Arrays.asList(text.split("\n")), new Filter<String>() { 
        public boolean accept(String s) {
          return s.trim().length() > 0;
        }
        private static final long serialVersionUID = 1;
      });
    int numRows = lines.size();
    int numCols = lines.get(0).trim().split("\\s+").length;
    double[][] data = new double[numRows][numCols];
    for (int row = 0; row < numRows; ++row) {
      String line = lines.get(row);
      String[] pieces = line.trim().split("\\s+");
      if (pieces.length != numCols) {
        throw new RuntimeException("Unexpected row length in line " + row);
      }
      for (int col = 0; col < numCols; ++col) {
        data[row][col] = Double.valueOf(pieces[col]);
      }
    }
    return new SimpleMatrix(data);
  }


  /**
   * Compute cosine distance between two column vectors.
   */
  public static double cosine(SimpleMatrix vector1, SimpleMatrix vector2){
    return dot(vector1, vector2)/(vector1.normF()*vector2.normF());
  }
  
  /**
   * Compute dot product between two vectors.
   */
  public static double dot(SimpleMatrix vector1, SimpleMatrix vector2){
    double score = Double.NaN;
    if(vector1.numRows()==1){ // vector1: row vector, assume that vector2 is a row vector too 
      score = vector1.mult(vector2.transpose()).get(0); 
    } else if (vector1.numCols()==1){ // vector1: col vector, assume that vector2 is also a column vector.
      score = vector1.transpose().mult(vector2).get(0);
    } else {
      System.err.println("! Error in neural.Utils.dot: vector1 is a matrix " + vector1.numRows() + " x " + vector1.numCols());
      System.exit(1);
    }

    return score;
  }
  
  /**
   * Given a sequence of Iterators over SimpleMatrix, fill in all of
   * the matrices with the entries in the theta vector.  Errors are
   * thrown if the theta vector does not exactly fill the matrices.
   */
  public static void vectorToParams(double[] theta, Iterator<SimpleMatrix> ... matrices) {
    int index = 0;
    for (Iterator<SimpleMatrix> matrixIterator : matrices) {
      while (matrixIterator.hasNext()) {
        SimpleMatrix matrix = matrixIterator.next();
        int numElements = matrix.getNumElements();
        for (int i = 0; i < numElements; ++i) {
          matrix.set(i, theta[index]);
          ++index;
        }
      }
    }
    if (index != theta.length) {
      throw new AssertionError("Did not entirely use the theta vector");
    }
  }

  /**
   * Given a sequence of iterators over the matrices, builds a vector
   * out of those matrices in the order given.  Asks for an expected
   * total size as a time savings.  AssertionError thrown if the
   * vector sizes do not exactly match.
   */
  public static double[] paramsToVector(int totalSize, Iterator<SimpleMatrix> ... matrices) {
    double[] theta = new double[totalSize];
    int index = 0;
    for (Iterator<SimpleMatrix> matrixIterator : matrices) {
      while (matrixIterator.hasNext()) {
        SimpleMatrix matrix = matrixIterator.next();
        int numElements = matrix.getNumElements();
        //System.out.println(Integer.toString(numElements)); // to know what matrices are
        for (int i = 0; i < numElements; ++i) {
          theta[index] = matrix.get(i);
          ++index;
        }
      }
    }
    if (index != totalSize) {
      throw new AssertionError("Did not entirely fill the theta vector: expected " + totalSize + " used " + index);
    }
    return theta;
  }

  /**
   * Given a sequence of iterators over the matrices, builds a vector
   * out of those matrices in the order given.  The vector is scaled
   * according to the <code>scale</code> parameter.  Asks for an
   * expected total size as a time savings.  AssertionError thrown if
   * the vector sizes do not exactly match.
   */
  public static double[] paramsToVector(double scale, int totalSize, Iterator<SimpleMatrix> ... matrices) {
    double[] theta = new double[totalSize];
    int index = 0;
    for (Iterator<SimpleMatrix> matrixIterator : matrices) {
      while (matrixIterator.hasNext()) {
        SimpleMatrix matrix = matrixIterator.next();
        int numElements = matrix.getNumElements();
        for (int i = 0; i < numElements; ++i) {
          theta[index] = matrix.get(i) * scale;
          ++index;
        }
      }
    }
    if (index != totalSize) {
      throw new AssertionError("Did not entirely fill the theta vector: expected " + totalSize + " used " + index);
    }
    return theta;
  }

  /**
   * Returns a sigmoid applied to the input <code>x</code>.
   */
  public static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }
  
  /**
   * Elementwise apply sigmoid.
   */
   public static SimpleMatrix elementwiseApplySigmoid(SimpleMatrix input) {
       SimpleMatrix output = new SimpleMatrix(input);
       for (int i = 0; i < output.numRows(); ++i) {
         for (int j = 0; j < output.numCols(); ++j) {
           output.set(i, j, sigmoid(output.get(i, j)));
         }
       }
       return output;
   }

  /**
   * Applies softmax to all of the elements of the matrix.  The return
   * matrix will have all of its elements sum to 1.  If your matrix is
   * not already a vector, be sure this is what you actually want.
   */
  public static SimpleMatrix softmax(SimpleMatrix input) {
    SimpleMatrix output = new SimpleMatrix(input);
    for (int i = 0; i < output.numRows(); ++i) {
      for (int j = 0; j < output.numCols(); ++j) {
        output.set(i, j, Math.exp(output.get(i,j)));
      }
    }
    double sum = output.elementSum();
    // will be safe, since exp should never return 0
    return output.scale(1.0 / sum); 
  }

  /**
   * Applies log to each of the entries in the matrix.  Returns a new matrix.
   */
  public static SimpleMatrix elementwiseApplyLog(SimpleMatrix input) {
    SimpleMatrix output = new SimpleMatrix(input);
    for (int i = 0; i < output.numRows(); ++i) {
      for (int j = 0; j < output.numCols(); ++j) {
        output.set(i, j, Math.log(output.get(i, j)));
      }
    }
    return output;
  }

  /**
   * Applies tanh to each of the entries in the matrix.  Returns a new matrix.
   */
  public static SimpleMatrix elementwiseApplyTanh(SimpleMatrix input) {
    //return elementwiseFunkyFunc(input, 7);
    SimpleMatrix output = new SimpleMatrix(input);
    for (int i = 0; i < output.numRows(); ++i) {
      for (int j = 0; j < output.numCols(); ++j) {
        output.set(i, j, Math.tanh(output.get(i, j)));
      }
    }
    return output;
  }

  /**
   * Applies the derivative of tanh to each of the elements in the vector.  Returns a new matrix.
   */
  public static SimpleMatrix elementwiseApplyTanhDerivative(SimpleMatrix input) {
    //return elementwiseFunkyGrad(input, 7);
    SimpleMatrix output = new SimpleMatrix(input.numRows(), input.numCols());
    output.set(1.0);
    output = output.minus(input.elementMult(input));
    return output;
  }
  
  /**
   * Applies tanh-ish non-saturating nonlinearity. Returns a new matrix.
   * Note: the gradient of f(x) is 2^(-floor(abs(x))), with the distal tails
   * of the function (i.e. when floor(abs(x)) is more than final_kink) remaining
   * linear. This last bit is why this function doesn't "saturate". Given that
   * f(0) is 0, the slope determines f(x) for all x. The degree to which f
   * saturates can be controlled by choosing different values for final_kink.
   */
  public static SimpleMatrix elementwiseFunkyFunc(SimpleMatrix input, int final_kink) {
    if (final_kink < 1) {
        // Funktion should have at least one kink, right?
        final_kink = 1;
    } else if (final_kink > 7) {
        // NeuralUtils.funky[Slope|Start] only has so many entries...
        final_kink = 7;
    }
    int bottom;
    double in_val;
    double abs_val;
    double out_val;
    double [] fStart = NeuralUtils.funkyStart;
    double [] fSlope = NeuralUtils.funkySlope;
    SimpleMatrix output = new SimpleMatrix(input.numRows(), input.numCols());
    for (int i=0; i < input.numRows(); i++) {
      for (int j=0; j < input.numCols(); j++) {
        in_val = 2.0 * input.get(i, j);
        abs_val = Math.abs(in_val);
        bottom = (int) Math.floor(abs_val);
        if (bottom > final_kink) {
            bottom = final_kink;
        }
        out_val = 0.5 * (fStart[bottom] + (fSlope[bottom] * (abs_val - ((double) bottom))));
        if (in_val >= 0) {
          output.set(i, j, out_val);
        } else {
          output.set(i, j, -out_val);
        }
      }
    }
    return output;
  }

  /**
   * Gradiates tanh-ish non-saturating nonlinearity. Returns a new matrix.
   * Note: the gradient of f(x) is 2^(-floor(abs(x))), with the distal tails
   * of the function (i.e. when floor(abs(x)) is more than final_kink) remaining
   * linear. This last bit is why this function doesn't "saturate".
   */
  public static SimpleMatrix elementwiseFunkyGrad(SimpleMatrix input, int final_kink) {
    if (final_kink < 0) {
        // Funktion should have at least one kink, right?
        final_kink = 0;
    } else if (final_kink > 7) {
        // NeuralUtils.funky[Slope|Start] only has so many entries...
        final_kink = 7;
    }
    double[] kink_val = NeuralUtils.funkyStart;
    double[] kink_slope = NeuralUtils.funkySlope;
    double funky_val;
    int funky_idx;
    SimpleMatrix output = new SimpleMatrix(input.numRows(), input.numCols());
    for (int i=0; i < input.numRows(); i++) {
      for (int j=0; j < input.numCols(); j++) {
          funky_idx = final_kink;
          funky_val = Math.abs(2.0 * input.get(i, j));
          for (int k=0; k < final_kink; k++) {
              if (funky_val > kink_val[k]) {
                  funky_idx = k;
              }
          }
          output.set(i, j, kink_slope[funky_idx]);
      }
    }
    return output;
  }

  /**
   * Concatenates several column vectors into one large column
   * vector, adds a 1.0 at the end as a bias term
   */
  public static SimpleMatrix concatenateWithBias(SimpleMatrix ... vectors) {
    int size = 0;
    for (SimpleMatrix vector : vectors) {
      size += vector.numRows();
    }
    // one extra for the bias
    size++;

    SimpleMatrix result = new SimpleMatrix(size, 1);
    int index = 0;
    for (SimpleMatrix vector : vectors) {
      result.insertIntoThis(index, 0, vector);
      index += vector.numRows();
    }
    result.set(index, 0, 1.0);
    return result;
  }

  /**
   * Concatenates several column vectors into one large column vector
   */
  public static SimpleMatrix concatenate(SimpleMatrix ... vectors) {
    int size = 0;
    for (SimpleMatrix vector : vectors) {
      size += vector.numRows();
    }

    SimpleMatrix result = new SimpleMatrix(size, 1);
    int index = 0;
    for (SimpleMatrix vector : vectors) {
      result.insertIntoThis(index, 0, vector);
      index += vector.numRows();
    }
    return result;
  }

  /**
   * Returns a vector with random Gaussian values, mean 0, std 1
   */
  public static SimpleMatrix randomGaussian(int numRows, int numCols, Random rand) {
    SimpleMatrix result = new SimpleMatrix(numRows, numCols);
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        result.set(i, j, rand.nextGaussian());
      }
    }
    return result;
  }
}

