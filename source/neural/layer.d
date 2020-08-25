module neural.layer;

import neural.tensor;
import neural.activation;
import neural.optimizer;

import std.random;
import std.math;

/// Base abstract class of all types of layers.
class NeuralLayer
{
    this()
    {
    }

    /// Input shape of this layer.
    final Shape inputShape() pure const nothrow @nogc
    {
        assert(_inputShape.isValid);
        return _inputShape;
    }
    
    /// Output dimension of this layer.
    final Shape outputShape() pure const nothrow @nogc
    {
        assert(_outputShape.isValid);
        return _outputShape;
    }    

    void predict(ref const(Tensor) input, ref Tensor output)
    {
        // Lazy initialization.
        if (!_initialized)
        {
            initialize(input.shape);
            _initialized = true;
        }

        // Eventually the output tensor might be uninitialized
        output.resize(_outputShape);

        assert(_inputShape == input.shape);
        assert(_outputShape == output.shape);
        doPredict(input, output);
    }

    // Tasks of this function
    // - should fill _inputShape and _outputShape.
    // - should allocate and initialize weights arrays.
    abstract void initialize(Shape inputShape);

    /// Returns: `true` if this layer has any parameters to be learnt.
    abstract bool isTrainable();

    // Forward inference
    abstract void doPredict(ref const(Tensor) input, ref Tensor output);

protected:
    bool _initialized = false;
    Shape _inputShape = invalidShape;
    Shape _outputShape = invalidShape;
}

class Sequential : NeuralLayer
{
    this()
    {
        super();
    }

    // You can add a sub-neural network
    void add(NeuralLayer layer, Shape inputShape = invalidShape)
    {
        if (inputShape.isValid)
        {
            // giving a shape is only valid for the first layer
            assert(_layers.length == 0);
            _currentInputShape = inputShape; 
        }

        _layers ~= layer;

        if (_currentInputShape.isValid)
        {
            layer.initialize(_currentInputShape);
            _currentInputShape = layer.outputShape;
        }
    }

    override void initialize(Shape inputShape)
    {
        _inputShape = inputShape;

        Shape shape = inputShape;        
        foreach(layer; _layers)
        {
            layer.initialize(shape);
            shape = layer.outputShape;
        }
    }

    override bool isTrainable()
    {
        foreach(layer; _layers)
        {
            if (layer.isTrainable)
                return true;
        }
        return false;
    }

    override void doPredict(ref const(Tensor) input, ref Tensor output)
    {
        Tensor t;
        tensorAssign(t, input);
        foreach(layer; _layers)
        {
            layer.predict(t, output);
            tensorAssign(t, output);
        }
    }

    /// Gets the list of layers.
    inout(NeuralLayer)[] layers() inout
    {
        return _layers;
    }

    /// Drops last layer.
    void pop()
    {
        _layers = _layers[0..$-1];
    }

    void compile(Optimizer optimizer, LossFunction loss)
    {
        _optimizer = optimizer;
        _lossFunction = loss;
    }

    // TODO: x and y should be tensors... but dimensionality would be broken
    void fit(float[] x, float[] y, int epochs)
    {
        foreach(epoch; 0..epochs)
        {
         /*   foreach()
            // forward pass
*/

        }

    }


private:
    NeuralLayer[] _layers;

    // used while building the net, for eager initialization
    Shape _currentInputShape = invalidShape; 

    // Should be here??   
    Optimizer _optimizer;
    LossFunction _lossFunction;
}

/// Input layer  (not sure why useful in keras)
class Input : NeuralLayer
{
    this(Shape shape)
    {

    }

    override void initialize(Shape inputShape)
    {
        _inputShape = inputShape;
        _outputShape = inputShape;
    }

    override bool isTrainable()
    {
        return false;
    }

    override void doPredict(ref const(Tensor) input, ref Tensor output)
    {
        assert(false);
    }
}



/// Dense, linear layer.
class Dense : NeuralLayer
{
    this(int batchSize)
    {
        super();
        _batchSize = batchSize;

      //  _bias.length = dimensionOut;
      //  _weight.length = dimensionOut * dimensionIn;
    }

    override void initialize(Shape inputShape)
    {
        assert(inputShape.is1D);

        _inputShape = inputShape;
        _outputShape = inputShape.withBatchSize(_batchSize);

        // Initialize bias
        _bias.length = _batchSize;
        _bias[] = 0;

        // Initialize weights
        _weight.length = _batchSize * inputShape.batchSize;

        // Xavier initialization for the weights
        float upperBound = sqrt(6.0f) / sqrt(cast(float)_inputShape.batchSize + _outputShape.batchSize);
        float lowerBound = -upperBound;

        foreach(ref w; _weight)
        {
            w = uniform(lowerBound, upperBound);
        }
    }

    override bool isTrainable()
    {
        return true;
    }

    override void doPredict(ref const(Tensor) input, ref Tensor output)
    {
        int numIns = input.shape().batchSize;
        int numOuts = output.shape().batchSize;

        // For now, only 1x1x1 images supported
        assert(input.shape.is1D && output.shape.is1D);

        for (int nOut = 0; nOut < numOuts; ++nOut)
        {
            float sum = _bias[nOut];
            for (int nIns = 0; nIns < numIns; ++nIns)
            {
                sum += input.rawData[nIns] * _weight[nIns + nOut * numIns];
            }
            output.rawData[nOut] = sum;
        }        
    }

private:
    float[] _bias; // _numOuts biases
    float[] _weight; // _numIns * _numOuts weights

    int _batchSize;
}

unittest
{
    auto layer = new neural.layer.Dense(32);
    //auto input = randomUniform(Shape(10, 20));

    auto input = randomUniform(Shape(10));
    Tensor output;
    layer.predict(input, output);
}

unittest
{
    // Optionally, the first layer can receive an input shape argument.
    auto model = new Sequential;
    model.add(new Dense(8), Shape(16));
}

unittest
{
    auto model = new Sequential;
    model.add(new Input(Shape(16)));
    model.add(new Dense(8));
}

// Do other things in https://keras.io/api/models/sequential/


/// Non-linear layer
class Activation : NeuralLayer
{
    this(ActivationFunction activationFunction)
    {
        super();
        _activationFunction = activationFunction;
    }

    override void initialize(Shape inputShape)
    {
        _inputShape = inputShape;
        _outputShape = inputShape;
    }

    override bool isTrainable()
    {
        return false;
    }

    override void doPredict(ref const(Tensor) input, ref Tensor output)
    {
        tensorCopy(output, input);        
        applyActivationFunction(_activationFunction, output.rawData);
    }

private:
    ActivationFunction _activationFunction;
}
