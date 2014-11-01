package edu.stanford.nlp.neural;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

/**
 * This class defines a block tensor, somewhat like a three
 * dimensional matrix.  This can be created in various ways, such as
 * by providing an array of SimpleMatrix slices, by providing the
 * initial size to create a 0-initialized tensor, or by creating a
 * random matrix.
 *
 * @author John Bauer
 * @author Richard Socher
 */
public class SimpleTensor implements Serializable {
  private SimpleMatrix[] slices;

  final int numRows;
  final int numCols;
  final int numSlices;

  /**
   * Creates a zero initialized tensor
   */
  public SimpleTensor(int numRows, int numCols, int numSlices) {
    slices = new SimpleMatrix[numSlices];
    for (int i = 0; i < numSlices; ++i) {
      slices[i] = new SimpleMatrix(numRows, numCols);
    }

    this.numRows = numRows;
    this.numCols = numCols;
    this.numSlices = numSlices;
  }

  /**
   * Copies the data in the slices.  Slices are copied rather than
   * reusing the original SimpleMatrix objects.  Each slice must be
   * the same size.
   */
  public SimpleTensor(SimpleMatrix[] slices) {
    this.numRows = slices[0].numRows();
    this.numCols = slices[0].numCols();
    this.numSlices = slices.length;
    this.slices = new SimpleMatrix[slices.length];
    for (int i = 0; i < numSlices; ++i) {
      if (slices[i].numRows() != numRows || slices[i].numCols() != numCols) {
        throw new IllegalArgumentException("Slice " + i + " has matrix dimensions " + slices[i].numRows() + "," + slices[i].numCols() + ", expected " + numRows + "," + numCols);
      }
      this.slices[i] = new SimpleMatrix(slices[i]);
    }
    
  }

  /**
   * Returns a randomly initialized tensor with values draft from the
   * uniform distribution between minValue and maxValue.
   */
  public static SimpleTensor random(int numRows, int numCols, int numSlices, double minValue, double maxValue, java.util.Random rand) {
    SimpleTensor tensor = new SimpleTensor(numRows, numCols, numSlices);
    for (int i = 0; i < numSlices; ++i) {
      tensor.slices[i] = SimpleMatrix.random(numRows, numCols, minValue, maxValue, rand);
    }
    return tensor;
  }

  /**
   * Number of rows in the tensor
   */
  public int numRows() {
    return numRows;
  }

  /**
   * Number of columns in the tensor
   */
  public int numCols() {
    return numCols;
  }

  /**
   * Number of slices in the tensor
   */
  public int numSlices() {
    return numSlices;
  }

  /**
   * Total number of elements in the tensor
   */
  public int getNumElements() {
    return numRows * numCols * numSlices;
  }

  /**
   * Returns a new tensor which has the values of the original tensor
   * scaled by <code>scaling</code>.  The original object is
   * unaffected.
   */
  public SimpleTensor scale(double scaling) {
    SimpleTensor result = new SimpleTensor(numRows, numCols, numSlices);
    for (int slice = 0; slice < numSlices; ++slice) {
      result.slices[slice] = slices[slice].scale(scaling);
    }
    return result;
  }

  /**
   * Returns <code>other</code> added to the current object, which is unaffected.
   */
  public SimpleTensor plus(SimpleTensor other) {
    if (other.numRows != numRows || other.numCols != numCols || other.numSlices != numSlices) {
      throw new IllegalArgumentException("Sizes of tensors do not match.  Our size: " + numRows + "," + numCols + "," + numSlices + "; other size " + other.numRows + "," + other.numCols + "," + other.numSlices);
    }
    SimpleTensor result = new SimpleTensor(numRows, numCols, numSlices);
    for (int i = 0; i < numSlices; ++i) {
      result.slices[i] = slices[i].plus(other.slices[i]);
    }
    return result;
  }

  /**
   * Performs elementwise multiplication on the tensors.  The original
   * objects are unaffected.
   */
  public SimpleTensor elementMult(SimpleTensor other) {
    if (other.numRows != numRows || other.numCols != numCols || other.numSlices != numSlices) {
      throw new IllegalArgumentException("Sizes of tensors do not match.  Our size: " + numRows + "," + numCols + "," + numSlices + "; other size " + other.numRows + "," + other.numCols + "," + other.numSlices);
    }
    SimpleTensor result = new SimpleTensor(numRows, numCols, numSlices);
    for (int i = 0; i < numSlices; ++i) {
      result.slices[i] = slices[i].elementMult(other.slices[i]);
    }
    return result;
  }

  /**
   * Returns the sum of all elements in the tensor.
   */
  public double elementSum() {
    double sum = 0.0;
    for (SimpleMatrix slice : slices) {
      sum += slice.elementSum();
    }
    return sum;
  }

  /**
   * Set each SimpleMatrix to those in newValues.
   */
  public void set(SimpleTensor newTensor) {
      if (numSlices != newTensor.numSlices) {
          throw new IllegalArgumentException("Mismatched tensor slice count.");
      }
      for (int i=0; i < numSlices; i++) {
          slices[i].set(newTensor.slices[i]);
      }
      return;
  }

  /**
   * Use the given <code>matrix</code> in place of <code>slice</code>.
   * Does not copy the <code>matrix</code>, but rather uses the actual object.
   */
  public void setSlice(int slice, SimpleMatrix matrix) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    if (matrix.numCols() != numCols) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.numCols() + " columns, tensor has " + numCols);
    }
    if (matrix.numRows() != numRows) {
      throw new IllegalArgumentException("Incompatible matrix size.  Has " + matrix.numRows() + " columns, tensor has " + numRows);
    }
    slices[slice] = matrix;
  }
  
  /**
   * Get a value from arbitrary location in this tensor.
   */
  public double get(int row, int col, int slice) {
      if (slice < 0 || slice >= numSlices) {
        throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
      }
      return slices[slice].get(row, col);
  }

  /**
   * Set a value at arbitrary location in this tensor.
   */
  public void set(int row, int col, int slice, double value) {
      if (slice < 0 || slice >= numSlices) {
        throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
      }
      slices[slice].set(row, col, value);
      return;
  }

  /**
   * Returns the SimpleMatrix at <code>slice</code>.
   * <br>
   * The actual slice is returned - do not alter this unless you know what you are doing.
   */
  public SimpleMatrix getSlice(int slice) {
    if (slice < 0 || slice >= numSlices) {
      throw new IllegalArgumentException("Unexpected slice number " + slice + " for tensor with " + numSlices + " slices");
    }
    return slices[slice];
  }

  /**
   * Returns a column vector where each entry is the nth bilinear
   * product of the nth slices of the two tensors.
   */
  public SimpleMatrix bilinearProducts(SimpleMatrix in) {
    if (in.numCols() != 1) {
      throw new AssertionError("Expected a column vector");
    }
    if (in.numRows() != numCols) {
      throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
    }
    if (numRows != numCols) {
      throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
    }
    SimpleMatrix inT = in.transpose();
    SimpleMatrix out = new SimpleMatrix(numSlices, 1);
    for (int slice = 0; slice < numSlices; ++slice) {
      double result = inT.mult(slices[slice]).mult(in).get(0);
      out.set(slice, result);
    }
    return out;
  }

  /**
   * Returns a column vector where each entry is the nth bilinear product of
   * the given left and right vector with the nth slice of this tensor.
   */
  public SimpleMatrix myBilinearProducts(SimpleMatrix leftVector, SimpleMatrix rightVector) {
    if ((leftVector.numCols() != 1) || (rightVector.numCols() != 1)){
      throw new AssertionError("Expected a pair of column vectors");
    }
    if (leftVector.numRows() != numRows) {
      throw new AssertionError("leftVector mismatched with tensor row count.");
    }
    if (rightVector.numRows() != numCols) {
      throw new AssertionError("rightVector mismatched with tensor col count.");
    }
    SimpleMatrix leftTranspose = leftVector.transpose();
    SimpleMatrix out = new SimpleMatrix(numSlices, 1);
    for (int slice = 0; slice < numSlices; ++slice) {
      double result = leftTranspose.mult(slices[slice]).mult(rightVector).get(0);
      out.set(slice, result);
    }
    return out;
  }

  /**
   * Returns a column vector where each entry is the nth bilinear product of
   * the given left and right vector with the nth slice of this tensor, while
   * skipping computations according to some (dropout) mask.
   */
  public SimpleMatrix maskedBilinearProducts(SimpleMatrix leftVector, SimpleMatrix rightVector, SimpleMatrix maskVector) {
    if ((leftVector.numCols() != 1) || (rightVector.numCols() != 1)){
      throw new AssertionError("Expected a pair of column vectors");
    }
    if (leftVector.numRows() != numRows) {
      throw new AssertionError("leftVector mismatched with tensor row count.");
    }
    if (rightVector.numRows() != numCols) {
      throw new AssertionError("rightVector mismatched with tensor col count.");
    }
    SimpleMatrix leftTranspose = leftVector.transpose();
    SimpleMatrix out = new SimpleMatrix(numSlices, 1);
    double result = 0.0;
    for (int slice = 0; slice < numSlices; ++slice) {
      if (maskVector.get(slice) > 0.1) {
        result = leftTranspose.mult(slices[slice]).mult(rightVector).get(0);
        out.set(slice, result);
      }
    }
    return out;
  }

  /**
   * Returns an iterator over the <code>SimpleMatrix</code> objects contained in the tensor.
   */
  public Iterator<SimpleMatrix> iteratorSimpleMatrix() {
    return Arrays.asList(slices).iterator();
  }

  public static SimpleTensor randomGaussian(int numRows, int numCols, int numSlices, Random rand) {
    SimpleTensor result = new SimpleTensor(numRows, numCols, numSlices);
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
          for (int k = 0; k < numSlices; ++k) {
            result.set(i, j, k, rand.nextGaussian());
          }
      }
    }
    return result;
  }


  /**
   * Returns an Iterator which returns the SimpleMatrices represented
   * by an Iterator over tensors.  This is useful for if you want to
   * perform some operation on each of the SimpleMatrix slices, such
   * as turning them into a parameter stack.
   */
  public static Iterator<SimpleMatrix> iteratorSimpleMatrix(Iterator<SimpleTensor> tensors) {
    return new SimpleMatrixIteratorWrapper(tensors);
  }

  private static class SimpleMatrixIteratorWrapper implements Iterator<SimpleMatrix> {
    Iterator<SimpleTensor> tensors;
    Iterator<SimpleMatrix> currentIterator;

    public SimpleMatrixIteratorWrapper(Iterator<SimpleTensor> tensors) {
      this.tensors = tensors;
      advanceIterator();
    }

    public boolean hasNext() {
      if (currentIterator == null) {
        return false;
      }
      if (currentIterator.hasNext()) {
        return true;
      }
      advanceIterator();
      return (currentIterator != null);
    }

    public SimpleMatrix next() {
      if (currentIterator != null && currentIterator.hasNext()) {
        return currentIterator.next();
      }
      advanceIterator();
      if (currentIterator != null) {
        return currentIterator.next();
      }
      throw new NoSuchElementException();
    }

    private void advanceIterator() {
      if (currentIterator != null && currentIterator.hasNext()) {
        return;
      }
      while (tensors.hasNext()) {
        currentIterator = tensors.next().iteratorSimpleMatrix();
        if (currentIterator.hasNext()) {
          return;
        }
      }
      currentIterator = null;
    }

    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private static final long serialVersionUID = 1;
}
