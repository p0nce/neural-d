module neural.layer;


enum ActivationFunction
{
    ReLU,    
}


/// Base abstract class of all types of layers.
class NeuralLayer
{
    this(int inputDimension, int outputDimension, bool isLearnable)
    {
        _numIns  = inputDimension;
        _numOuts = outputDimension;
        _isLearnable = isLearnable;
    }

    /// Input dimension of this layer.
    final int numIns() pure const nothrow @nogc
    {
        return _numIns;
    }
    
    /// Output dimension of this layer.
    final int numOuts() pure const nothrow @nogc
    {
        return _numOuts;
    }

    /// Returns: `true` if this layer has any parameters to be learnt.
    final bool isTrainable()
    {
        return _isLearnable;
    }

    // Forward inference
    abstract void predict(const(float)[] inputs, float[] outputs);

protected:
    immutable(int) _numIns;
    immutable(int) _numOuts;
    bool _isLearnable;
}

// A NeuralNetwork can contain several layers, and is also a layer itself.
class NeuralNetwork : NeuralLayer
{
    this(int dimensionIn, int dimensionOut, ActivationFunction activation)
    {
        super(dimensionIn, dimensionOut, true);
    }

    void addLayer(NeuralLayer layer)
    {
        _layers ~= layer;
    }

    override void predict(const(float)[] inputs, float[] outputs)
    {
        // find max dimension in layers,
        // eventually extend scratch buffers
        int maxDimension = cast(int)inputs.length;
        if (maxDimension < cast(int)(outputs.length))
            maxDimension = cast(int)(outputs.length);
        foreach(layer; _layers)
        {
            if (layer.numIns() > maxDimension)
                maxDimension = numIns();
            if (layer.numOuts() > maxDimension)
                maxDimension = numOuts();
        }
        if (_temp1.length < maxDimension)
        {
            _temp1.length = maxDimension;
            _temp2.length = maxDimension;
        }

        _temp1[0..inputs.length] = inputs[];
        foreach(layer; _layers)
        {
            int nIns = layer.numIns();
            int nOuts = layer.numOuts();
            layer.predict(_temp1[0..nIns], _temp2[0..nOuts]);
            float[] exch = _temp1;
            _temp1 = _temp2;
            _temp2 = exch;
        }
        outputs[] = _temp1[0..outputs.length];
    }

private:
    float[] _temp1, _temp2;
    NeuralLayer[] _layers;
}

class LayerLinear : NeuralLayer
{
    this(int dimensionIn, int dimensionOut, ActivationFunction activation)
    {
        super(dimensionIn, dimensionOut, true);

        _bias.length = dimensionOut;
        _weight.length = dimensionOut * dimensionIn;
        _activation = activation;
    }

    override void predict(const(float)[] inputs, float[] outputs)
    {   
        assert(inputs.length == numIns()); 
        assert(outputs.length == numOuts());

        for (int nOut = 0; nOut < numOuts(); ++nOut)
        {
            float sum = _bias[nOut];
            for (int nIns = 0; nIns < numIns(); ++nIns)
            {
                sum += inputs[nIns] * _weight[nIns + nOut * numIns()];
            }
            outputs[nOut] = sum;
        }
        applyActivationFunction(outputs[]);
    }

private:
    ActivationFunction _activation;
    float[] _bias; // _numOuts biases
    float[] _weight; // _numIns * _numOuts weights

    void applyActivationFunction(float[] outputs)
    {
        final switch(_activation) with (ActivationFunction)
        {
            case ReLU:
                foreach(ref x; outputs)
                {
                    if (x < 0) x = 0;
                }
                break;
        }
    }
}


double nextRandom(ref ulong seed)
{
    seed = seed * 1584831133568692850 + 8629370730060374025;
    ulong r = seed >> 32;
    seed ^= r;
    return( ( cast(uint)r ) * 2.32830643653869629e-10 );
}