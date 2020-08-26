/*
 * Copyright: 2020 Guillaume Piolat.
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module neural.layer;

import neural.tensor;
import neural.activation;
import neural.optimizer;

import std.random;
import std.math;
import std.string;

/// Base abstract class of all types of layers.
class NeuralLayer
{
    this()
    {
    }

    /// Input shape of this layer.
    /// Layout of the input tensor is:
    /// 
    /// [input batchsize x image height x image width x color dimension]
    final Shape inputShape() pure const nothrow @nogc
    {
        assert(_inputShape.isValid);
        return _inputShape;
    }
    
    /// Output dimension of this layer.
    /// [output batchsize x image height x image width x color dimension]
    final Shape outputShape() pure const nothrow @nogc
    {
        assert(_outputShape.isValid);
        return _outputShape;
    }    

    void lazyInitialization(Shape inputShape)
    {
        // Lazy initialization.
        if (!_initialized)
        {
            initialize(inputShape);
            _initialized = true;
        }
        else
        {
            assert(inputShape == _inputShape);
        }
    }

    void predict(ref const(Tensor) input, ref Tensor output)
    {
        lazyInitialization(input.shape);        

        // Eventually the output tensor might be uninitialized
        output.resize(_outputShape);

        assert(_inputShape == input.shape);
        assert(_outputShape == output.shape);
        doPredict(input, output);
    }

    // TODO: should be a tensor of grads
    void accumulateGradient(float[] grad)
    {
        assert(grad.length == outputShape.dimension[0]);
        doAccumulateGradient(grad);
    }

    // Tasks of this function
    // - should fill _inputShape and _outputShape.
    // - should allocate and initialize weights arrays.
    // [input batchsize x image height x image width x color dimension]
    abstract void initialize(Shape inputShape);

    /// Returns: number of trainable parameters.
    abstract int trainableParams();

    /// Returns: `true` if this layer has any parameters to be learnt.
    abstract bool isTrainable();

    // Forward inference for one item.
    // The expected layout of the input and output tensor is:
    // [input batchsize x image height x image width x color dimension]
    abstract void doPredict(ref const(Tensor) input, ref Tensor output);

    /// `grad` contains cost gradients for each output of this layer?
    /// This function's goal is to fill the `_backGradients` member.
    // TODO: should be a tensor of grads
    abstract void doAccumulateGradient(float[] grad);

    /// Allocate and training-specific data structures.
    void allocateTrainingData()
    {
        assert(_initialized);
        _backGradients.length = inputShape.dimension[0];
    }

protected:
    bool _initialized = false;
    Shape _inputShape = invalidShape;
    Shape _outputShape = invalidShape;
    float[] _backGradients; // gradients to propagate back to previous layer
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

    override int trainableParams()
    {
        return 0;
    }

    override void doPredict(ref const(Tensor) input, ref Tensor output)
    {
        assert(false);
    }
}



/// Dense, linear layer.
class Dense : NeuralLayer
{
    this(int units)
    {
        super();
        _units = units;
    }

    override void initialize(Shape inputShape)
    {
        _inputUnits = inputShape.dimension[0];

        _inputShape = inputShape;
        _outputShape = inputShape;
        _outputShape.dimension[0] = _units;

        // Initialize bias
        _bias.length = _units;
        _bias[] = 0;

        // Initialize weights
        _weight.length = _units * _inputUnits;

        // Xavier initialization for the weights
        float upperBound = sqrt(6.0f) / sqrt(cast(float)_inputUnits + _units);
        float lowerBound = -upperBound;

        foreach(ref w; _weight)
        {
            w = uniform(lowerBound, upperBound);
        }
    }

    override void allocateTrainingData()
    {
        super.allocateTrainingData();
        _grad.length = _units * _inputUnits;
    }

    override int trainableParams()
    {
        assert(_initialized);
        return _units // bias
              + _units * _inputUnits; // weights
    }

    override bool isTrainable()
    {
        return true;
    }

    override void doPredict(ref const(Tensor) input, ref Tensor output)
    {
        int numIns = input.shape().dimension[0];
        int numOuts = output.shape().dimension[0];

        // For now, only 1x1x1 images supported, with N input fans
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

    override void doAccumulateGradient(const(float)[] grad)
    {
        for(int n = 0; n < grad.length; ++n)
        {
            float value = _lastActivationValues[n];
            _backGradients[n] = evalActivationFunctionDerivative(_activationFunction, value);
        }
    }

private:
    float[] _bias;   // _numOuts biases
    float[] _weight; // _numIns * _numOuts weights

    float[] _grad;

    int _units;
    int _inputUnits;
}

unittest
{
    auto layer = new neural.layer.Dense(32);
    //auto input = randomUniform(Shape(10, 20));

    auto input = tensorRandomUniform(Shape(10));
    Tensor output;
    layer.predict(input, output);
}

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

        _lastActivationValues.length = inputShape.dimension[0];
    }

    override bool isTrainable()
    {
        return false;
    }

    override int trainableParams()
    {
        return 0;
    }    

    override void doPredict(ref const(Tensor) input, ref Tensor output)
    {
        tensorCopy(output, input);        
        applyActivationFunction(_activationFunction, output.rawData);

        // TODO make this a tensor
        assert(output.shape.is1D);
        _lastActivationValues[] = output.rawData[];
    }

    override void doAccumulateGradient(const(float)[] grad)
    {
        for(int n = 0; n < grad.length; ++n)
        {
            float value = _lastActivationValues[n];
            _backGradients[n] = evalActivationFunctionDerivative(_activationFunction, value);
        }
    }

private:
    ActivationFunction _activationFunction;
    float[] _lastActivationValues;
}
