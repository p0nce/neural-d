module neural.model;

import std.string;

import neural.tensor;
import neural.layer;
import neural.optimizer;

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
            assert(!layer._initialized);
            layer.initialize(_currentInputShape);
            layer._initialized = true;
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

        _outputShape = shape; // same output shape as the last layer
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

    override int trainableParams()
    {
        int sum = 0;
        foreach(layer; _layers)
        {
            sum += layer.trainableParams();
        }
        return sum;
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

    void summary()
    {
        import std.stdio;

        writeln("Model:");
        writeln("_________________________________________________________________");
        writeln("Layer (type)                   Output Shape              Param #   ");
        writeln("=================================================================");

        foreach(layer; _layers)
        {
            string shapeStr = format("%s", layer.outputShape);
            writefln("%-30s %-26s %s",
                layer.classinfo.name, shapeStr, layer.trainableParams);
            writeln("_________________________________________________________________");
        }

        writefln("Total params: %s", trainableParams());
        writefln("Trainable params: %s", trainableParams());
        writeln("Non-trainable params: 0");
    }

    /// Train the neural network.
    /// The shape of tensors is:
    /// samples x units x image-height x image-width x colors
    void train(Tensor x, Tensor y, int epochs)
    {
        foreach(epoch; 0..epochs)
        {
            assert(x.shape.dimension[0] == y.shape.dimension[0]);
            int numSamples = x.shape.dimension[0];

            for (int sample = 0; sample < numSamples; ++sample)
            {
                Tensor subx = x[sample];
                Tensor suby = y[sample];

                // Forwards pass, get prediction
                Tensor pred;
                predict(subx, pred);


            }


         /*   foreach()
            // forward pass */

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