/*
 * Copyright: 2020 Guillaume Piolat.
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module neural.optimizer;

enum LossFunction
{
    MSE,
}

interface Optimizer
{    
    float learningRate();
}

class SGDOptimizer : Optimizer
{
    this(float learningRate = 0.01f)
    {
        _learningRate = learningRate;
    } 

    override float learningRate()
    {
        return _learningRate;
    }

private:
    float _learningRate;
}