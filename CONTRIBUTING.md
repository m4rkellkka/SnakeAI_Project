# Contributing to Snake AI

First off, thank you for considering contributing to Snake AI! It's people like you that make Snake AI such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](https://github.com/m4rkellkka/SnakeAI_Project/issues) to see if someone else has already created a ticket. If not, go ahead and make one!

## Fork & create a branch

If this is something you think you can fix, then fork Snake AI and create a branch with a descriptive name.

## Get the test suite running

Make sure you have installed all dependencies via `pip install -r requirements.txt`. 
You can run the main training loop or the benchmark tools to ensure everything is still functioning as expected.

```bash
python tools/benchmark.py --checkpoint model/pretrained.pth --games 5
```

## Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first.
Please ensure that your code follows the existing style, includes appropriate comments, and doesn't break the existing neural network architecture unless intended.

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with Snake AI's main branch.
Then create a Pull Request against our main branch using the provided PR template.
