/*
 * Copyright: 2020 Guillaume Piolat.
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
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
            layer.lazyInitialization(_currentInputShape);
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

    override void startBatch()
    {      
        foreach(layer; _layers)
            layer.startBatch();
    }

    override void doAccumulateGradient(ref const(Tensor) forwardGradients, ref Tensor backGradients)
    {
        Tensor t;
        tensorAssign(t, forwardGradients);
        foreach_reverse(layer; _layers)
        {
            layer.doAccumulateGradient(t, backGradients);
            tensorAssign(t, backGradients);
        }
    }    

    override void stopBatch(float learningRate)
    {      
        foreach(layer; _layers)
            layer.stopBatch(learningRate);
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
    void train(Tensor x, Tensor y, int minibatchSize, int epochs)
    {
        int numSamples = x.shape.dimension[0];        
        assert(numSamples >= minibatchSize); // else, no learning

        int numBatches = numSamples / minibatchSize;

        // Initialize the network, so that every layer has a defined shape.
        lazyInitialization(x[0].shape);

        foreach(epoch; 0..epochs)
        {
            assert(x.shape.dimension[0] == y.shape.dimension[0]);   

            for (int batch = 0;  batch < numBatches; ++batch)
            {
                int batchStart = batch * minibatchSize;         
                int batchStop = (batch+1) * minibatchSize;

                // Start a mini-batch

                double totalLoss = 0;

                startBatch();

                for (int sample = batchStart; sample < batchStop; ++sample)
                {
                    Tensor subx = x[sample];
                    Tensor suby = y[sample];

                    // Forwards pass, get prediction
                    Tensor pred;
                    predict(subx, pred);


                    // Compute output gradient difference with regards to 
                    // the MSE error.
                    // dMSE/pred = 2*(pred - suby)
                    Tensor grad;
                    tensorAssign(grad, pred);
                    tensorSub(grad, suby);

//import std.stdio;
  //                  writefln("%s", grad.rawData);


                    // Back propagate gradient.
                    Tensor gradients, backGradients;
                    tensorAssign(gradients, grad);
                    foreach_reverse(layer; _layers)
                    {
                        layer.accumulateGradient(gradients, backGradients);
                        tensorAssign(gradients, backGradients);
                    }
                }

                float lr = _optimizer.learningRate();
                stopBatch(lr);
            }
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