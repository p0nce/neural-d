module neural.layer;


/// Base abstract class of all types of layers.
class NeuralLayer
{
    this(int inputDimension, int outputDimension)
    {
        _numIns  = inputDimension;
        _numOuts = outputDimension;
    }

    /// Input dimension of this layer.
    int numIns() pure const nothrow @nogc
    {
        return _numIns;
    }
    
    /// Output dimension of this layer.
    int numOuts() pure const nothrow @nogc
    {
        return _numOuts;
    }

    /// Returns: `true` if this layer has any parameters to be learnt.
    abstract bool isTrainable();

    // Forward inference
    abstract void predict(immutable(float)[] inputs, float[] outputs);

protected:
    immutable(int) _numIns;
    immutable(int) _numOuts;
}

class LayerDense : NeuralLayer
{
    this(int dimensionIn, int dimensionOut)
    {
        super(dimensionIn, dimensionOut);

        _bias.length = dimensionOut;
        _weight.length = dimensionOut * dimensionIn;
    }

    /// Returns: `true` if this layer has any parameters to be learnt.
    override bool isTrainable()
    {
        return true;
    }

    override void predict(immutable(float)[] inputs, float[] outputs)
    {        
        for (int nOut = 0; nOut < numOuts(); ++nOut)
        {
            float sum = _bias[nOut];
            for (int nIns = 0; nIns < numIns(); ++nIns)
            {
                sum += inputs[nIns] * _weight[nIns + nOut * numIns()];
            }
            outputs[nOut] = sum;
        }
    }
private:
    float[] _bias; // _numOuts biases
    float[] _weight; // _numIns * _numOuts weights

}

