module neural.optimizer;

enum LossFunction
{
    MSE,
}

interface Optimizer
{    
}

class SGDOptimizer : Optimizer
{
    this(float learningRate = 0.01f)
    {
        _learningRate = learningRate;
    } 

private:
    float _learningRate;
}