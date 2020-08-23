module neural.network;

class Model
{
    void addLayer(Layer layer)
    {
        _layers ~= layer;
    }

    Layer getLayer(int n)
    {
        return _layers[n];
    }

private:
    Layer[] _layers; 
}

enum ActivationFunction
{
    relu,
    softmax,
    sigmoid
}

class Layer
{
    this(Layer parentLayer)
    {
        
    }
}