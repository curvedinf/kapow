# Kapow
Python metafunctions

## What is a metafunction?

A normal function has defined input types, defined output types, and deterministic 
results for its output. On the other hand, a metafunction is a type of function which 
has undefined input and undefined non-deterministic output. Where functions
give known outputs, metafunctions are designed to self optimize over time
and make no assumptions about the scope of their inputs or outputs.

You may ask why this would be useful. The answer is that a single
generic metafunction definition can approximate the functionality of all
normal functions, while also being able to adapt to a class of problems 
which normal functions are not well suited for: undefined scope problems,
which will be referred to USPs below.

### Andromeda and the Undefined Scope Problem

An easy way to understand why USPs are important is to imagine if you sent
an autonomous spacecraft to our closest neighboring galaxy, Andromeda. 
After traveling to Andromeda it will take 2.5 million years to communicate
with it again, so the spacecraft's programming is effectively unchangeable.

After it arrives, it finds that the laws of physics have wildly become backwards!
Up has become down, right has become left, and forward has become backwards.
This offers quite a conundrum for the spacecraft, as if it were created
with the assumption of normal physics being universal, it would not be able to
operate properly.

On the other hand, if the programmers of this spacecraft did not assume
anything about the laws of Andromeda they would be considering it an 
undefined scope problem. In this case, they could use metafunctions to
program craft.

If the spacecraft used metafunctions, once it began experiencing sub-optimal
performance from the new laws of physics, the functionality of its internal
programming would automatically self-optimize to the new laws. It could also
travel back to the Milky Way and immediately assume the laws of physics are
normal again.

### The Rabbit Hole of USPs Goes Deeper

In the example above we can understand how metafunctions have non-deterministic
output and why that is useful, but we have not yet explored why the ability to
handle undefined input and output is useful.

Let's go back to our spacecraft example to demonstrate. In exploring the
backwards galaxy Andromeda, the spacecraft comes across a planet that has
the signals of life, and as its imperrative is to seek out new life, it
must confirm the life signs.

In doing so it must decend into the atmosphere of tthe planet, where
it has determined all of its sensors will no longer function, and none of
its existing sources of motivation will work. To this end, it designs and
constructs new specialized sensors and motivators.

If its navigation were created with normal functions, they would not be able
to adapt to the new sensors and motivators of this USP. A smart programmer could expect
this scope of condition and design abstraction that accounts for it,
but what if it were impossible to expect the contingent conditions? For example, 
what if the world which the spacecraft found was a flatworld, with one 
less dimension?

Every metafunction accepts any scope, quantity, type, or range of input and
output variables. It also does not assume the input or output scopes
will be fixed over time. It accepts anything and does anything, with the
only limitation being whether there is an ability to define a more 
optimal set of outputs given any set of inputs.

## Using Kapow

Kapow is a library which implements a metafunction definition that can be used
in a familiar way to normal functions. It is suited for problems with highly
variable scope or considerable unknowns, such as rapid prototyping and
AI agents.

Here is a basic use case of a Kapow metafunction that estimates square roots:

```python
import random
import math

from kapow import mf

def optimal_sqrt(mf_output, args, kwargs):
    # mf_output is the output of the metafunction.
    # It would normally be used as the basis for optimization,
    # but in this simple example, we use an exact function to model.
    return math.sqrt(kwargs["input_number"])

@mf(optimal_sqrt)
def approximate_sqrt(input_number=float):
    return float

for _ in range(1000):
    random_int = random.randint(0, 100000)
    approx_sqrt = approximate_sqrt(random_int)
    actual_sqrt = math.sqrt(random_int)
    print(f"Square root of {random_int}:")
    print(f"Approx: {approx_sqrt}")
    print(f"Actual: {actual_sqrt}")
    print(f"Diff: {approx_sqrt-actual_sqrt}")
```

## Model Architecture

The model architecture is composed of an encoder, regression neural network, and
a decoder. The encoder encodes the input and encodes it into a json string,
which is then plotted into a latent space using a transformer model's encoder.
The transformer model's encoder produces an embedding vector that normalizes
the function inputs into latent space. The regression neural network then 
uses the embedding vector as input to regress an output embedding vector that
predicts the function output's location in latent space. Finally, the decoder 
uses a transformer model's decoder to predict a series of tokens for a json
encoded function output. The JSON is then decoded and the output is returned.

The model also self-trains using the optimal output function that is required
by the metafunction. The training is performed only on the regression neural
network by running the 