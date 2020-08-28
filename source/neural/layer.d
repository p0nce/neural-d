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

    /// Predict several samples at once (a tensor of input tensors).
    /// TODO: if predict can take any tensor size, then maybe there wouldn't be a need
    /// for such an operation.
    void predictBatch(ref const(Tensor) input, ref Tensor output)
    {
        lazyInitialization(input[0].shape);

        int sampleCount = input.shape.dimension[0];

        // Eventually the output tensor might be uninitialized
        output.resize(_outputShape.inLargerArray(sampleCount));
        Tensor outsample;
        for (int n = 0; n < sampleCount; n++)
        {
            Tensor insample = input[n];
            predict(insample, outsample);
            Tensor item = output[n];
            tensorCopy(item, outsample);
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

    /// `forwardGradients` is the gradient of error for the output
    /// `backGradients` is the gradient of error for the input, for backpropagation.
    /// If computing `inGradients` requires computing activation values, 
    /// those are to be 
    void accumulateGradient(ref const(Tensor) forwardGradients, ref Tensor backGradients)
    {
        // Eventually the backward tensor might be uninitialized
        backGradients.resize(_inputShape);
        assert(forwardGradients.shape == _outputShape);
        assert(backGradients.shape == _inputShape);
        doAccumulateGradient(forwardGradients, backGradients);
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

    /// Start training.
    /// `doAccumulateGradient` will be called a number of times between `startBatch()` and `stopBatch()`.
    /// `startBatch()` should allocate training structures if needed.
    void startBatch()
    {        
    }

    /// `forwardGradients` contains cost gradients for each output of this layer.
    /// This function's goal is to fill the `backGradients` member.
    abstract void doAccumulateGradient(ref const(Tensor) forwardGradients, ref Tensor backGradients);

    /// `stopBatch()` should train parameters based on the work performed between 
    /// `doAccumulateGradient`.
    /// mini-batch size expected to be counted by `doAccumulateGradient` in order to perform averages.
    void stopBatch(float learningRate)
    {        
    }


protected:
    bool _initialized = false;
    Shape _inputShape = invalidShape;
    Shape _outputShape = invalidShape;
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

    override int trainableParams()
    {
        assert(_initialized);
        return _units // biases
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
        // TODO: lift this limitation!
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

        tensorAssign(_lastInput, input);
    }

    override void startBatch()
    {
        _batchSize = 0;

        // Weight gradient reset
        _weightGrad.length = _units * _inputUnits;
        _weightGrad[] = 0;

        // Bias gradient reset
        _biasGrad.length = _units;
        _biasGrad[] = 0;
    }

    override void stopBatch(float learningRate)
    {
        // averages the learning rate by item in the mini-batch
        learningRate /= _batchSize;

        // learn weights
        _weight[] -= _weightGrad[] * learningRate;

        // learn biases
        _bias[]   -= _biasGrad[]   * learningRate;
    }

    override void doAccumulateGradient(ref const(Tensor) forwardGradients, ref Tensor backGradients)
    {
        int numIns = backGradients.shape().dimension[0];
        int numOuts = forwardGradients.shape().dimension[0];

         // For now, only 1x1x1 images supported, with N input fans
        assert(forwardGradients.shape.is1D && backGradients.shape.is1D);

        assert(_lastInput.shape.is1D);

        // Compute loss gradient, when derived by each trainable parameter.
        for (int nOut = 0; nOut < numOuts; ++nOut)
        {
            for (int nIns = 0; nIns < numIns; ++nIns)
            {
                _weightGrad[nIns + nOut * numIns] += forwardGradients.rawData[nOut] * _lastInput.rawData[nIns];
            }
            _biasGrad[nOut] += forwardGradients.rawData[nOut];
        }

        for (int nIns = 0; nIns < numIns; ++nIns)
        {
            backGradients.rawData[nIns] = 0.0f;
        }

        for (int nOut = 0; nOut < numOuts; ++nOut)
        {
            float sum = _bias[nOut];
            for (int nIns = 0; nIns < numIns; ++nIns)
            {
                backGradients.rawData[nIns] += forwardGradients.rawData[nOut] * _weight[nIns + nOut * numIns];
            }
        }

        _batchSize = _batchSize + 1;
    }

private:
    float[] _bias;   // _numOuts biases
    float[] _weight; // _numIns * _numOuts weights

    int _units;
    int _inputUnits;

    Tensor _lastInput;
    float[] _weightGrad;
    float[] _biasGrad;
    int _batchSize;
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

        _lastActivationValues.resize(inputShape);
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

        tensorCopy(_lastActivationValues, input);
    }

    override void doAccumulateGradient(ref const(Tensor) forwardGradients, ref Tensor backGradients)
    {
        int numScalars = _inputShape.elemCount();
        for(int n = 0; n < numScalars; ++n)
        {
            float value = _lastActivationValues.rawData[n];
            backGradients.rawData[n] = forwardGradients.rawData[n]
                                     * evalActivationFunctionDerivative(_activationFunction, value);
        }
    }

private:
    ActivationFunction _activationFunction;
    Tensor _lastActivationValues;
}
