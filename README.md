# Hellm-world

This repository serves as a project to learn LLM's from first principles for [me](https://its.beer). Please read the 'thank you's' section down below if you can for credits of sources that helped me.

## No vibe-coding or dependencies

The goal is to not use any AI-generated code or dependencies for the actual inference & training logic. Instead, I intend to write every single line myself; after fully understanding why it's written like that.

❌ Not allowed to be touched by coding agents:
* Inference logic
* Training logic
* LLM harnass

✅ allowed to be touched by coding agents:
* Unit tests
* Infrastructure scripts like chart generation

## JS-tradeoffs

I know JS is not optimal for heavy linear algebra and mathematics. But, I'm here to learn at the highest speed possible; and am most comfortable in JS. Thus the choice to write in this.

## Timmy - a 6k model

Guaranteed working version on tag `timmy` (`git checkout timmy`).

```
pnpm describe timmy

Parameter count: 6.6K
Transformer count: 2
Attention head count: 2
Hidden dimensions size: 16
```

I've written about the behavior and process of training the model here: [http://its.beer/thoughts/training-timmy](http://its.beer/thoughts/training-timmy).

![Training curve of Timmy](./model/timmy/checkpoint_000007_loss.png)

## Thank you's & learning resources

My main source of understanding is coming from the video series of [3Blue1Brown](https://www.youtube.com/@3blue1brown).

* Inference videos
  * [Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6)
  * [Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=7)
  * [Multilayer Perceptron](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8)
* Training videos
  * [Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)
  * Backprop [part 1](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3) & [part 2](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)

On top of that AI coding agents helped me out a bunch. They were not allowed to write any inference and training code or give me answers, but were instructed to through hints and TDD guide me to them.