module neural.tensor;

import std.random;

// Describe the size of a 1D to 5D array.
struct Shape
{
    int[5] dimension;

    this(int dim0, int dim1 = 1, int dim2 = 1, int dim3 = 1, int dim4 = 1)
    {
        dimension[0] = dim0;
        dimension[1] = dim1;
        dimension[2] = dim2;
        dimension[3] = dim3;
        dimension[4] = dim4;
    }

    // Returns: `true` if the shape is a valid shape. Else it's an unknown shape.
    bool isValid() pure const nothrow @nogc
    {
        // Example: 30 images of 25x25 RGB pixels are (30, 25, 25, 3)
        return (dimension[0]          >= 1) 
            && (dimension[1]          >= 1)
            && (dimension[2]          >= 1)
            && (dimension[3]          >= 1)
            && (dimension[4]          >= 1);
    }

    int numDimensions() pure const nothrow @nogc
    {
        if (dimension[4] > 1) return 5;
        if (dimension[3] > 1) return 4;
        if (dimension[2] > 1) return 3;
        if (dimension[1] > 1) return 2;
        if (dimension[0] > 1) return 1;
        return 0;
    }

    bool is1D() pure const nothrow @nogc { return numDimensions() == 1; }
    bool is2D() pure const nothrow @nogc { return numDimensions() == 1; }
    bool is3D() pure const nothrow @nogc { return numDimensions() == 1; }
    bool is4D() pure const nothrow @nogc { return numDimensions() == 1; }
    bool is5D() pure const nothrow @nogc { return numDimensions() == 1; }

    // Number of scalar number to represent this shape.
    int elemCount() pure const nothrow @nogc
    {
        return dimension[0]*dimension[1]*dimension[2]*dimension[3]*dimension[4];
    }
}

enum invalidShape = Shape(-1, -1, -1, -1, -1);

unittest
{
    assert(!invalidShape.isValid());
}

struct Tensor
{
public:
    this(Shape shape)
    {
        resize(shape);
    }

    this(float[] shape)
    {
        resize(Shape(cast(int)shape.length, 1, 1, 1));
        _data[] = shape[];
    }

    ~this()
    {
        _data.length = 0;
    }

    @disable this(this); // Use tensorCopy to make a copy

    void resize(Shape shape)
    {
        assert((_data is null));
        _data.length = shape.elemCount();
        _shape = shape;
    }

    Shape shape() const
    {
        return _shape;
    }

    void fillWith(float x)
    {
        _data[] = x;
    }

    inout(float)[] rawData() inout
    {
        return _data;
    }

    // Get a const tensor with less dimensions.
    const(Tensor) opIndex(size_t n)
    {
        assert(false);
    }

private:
    Shape _shape = Shape(0, 0, 0, 0);
    float[] _data = null;
}

unittest
{
    Tensor t;
    t.resize(Shape(5,2,3,4));
    const(Tensor) q = t[0];
}

Tensor tensorConstant(float value, Shape shape)
{
    Tensor t = Tensor(shape);
    t.fillWith(value);
    return t;
}

Tensor tensorZeroes(Shape shape)
{
    return tensorConstant(0.0f, shape);
}

Tensor tensorOnes(Shape shape)
{
    return tensorConstant(1.0f, shape);
}

Tensor tensorRandomUniform(Shape shape)
{
    Tensor t = Tensor(shape);
    foreach(ref f; t.rawData)
        f = uniform01();
    return t;
}

// Operations on tensor

/// Make a copy of the tensor. `dest` doesn't need to be pre-allocated.
void tensorAssign(ref Tensor dest, ref const(Tensor) source)
{
    dest.resize(source.shape);
    tensorCopy(dest, source);
}

/// Copy data of an existing allocated tensor into another allocated tensor.
void tensorCopy(ref Tensor dest, ref const(Tensor) source)
{
    assert(dest.shape == source.shape);
    dest._data[] = source._data[];
}


private
{
    @disable double nextRandom(ref ulong seed)
    {
        seed = seed * 1584831133568692850 + 8629370730060374025;
        ulong r = seed >> 32;
        seed ^= r;
        return( ( cast(uint)r ) * 2.32830643653869629e-10 );
    }
}

void axpy(float a, float[] x, float[] y) pure nothrow @nogc
{
    for(int i = 0; i < x.length; ++i)
        y[i] += a * x[i];
}