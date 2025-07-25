Rumi Deep Poetry Generator 
This project is a poetry generator trained on the works of Rumi. It uses a Transformer-based deep learning model to generate new poetic lines that resemble Rumi’s writing style. The model has been built from scratch using PyTorch and is enhanced with several advanced training techniques.

What This Project Does
Reads a dataset of Rumi's poetry

Trains a deep learning model to understand the structure, flow, and style of the poems

Generates new poetry character by character using that model

Tries to make the output rhyme and sound poetic

Key Features
Transformer Model: A neural network that pays attention to the context of previous words to decide the next one.

Custom Tokenizer: Breaks the poem into characters and turns them into numbers the model can understand.

Rhyme Detection: Helps the generator include words that rhyme, making the poem more musical.

Temperature Control: Lets you adjust how creative or predictable the model should be during generation.

Training Optimizations: Uses techniques like mixed precision (for speed), gradient clipping (for stability), and learning rate scheduling (for smoother training).

Top-k Sampling: The model chooses from only the top likely next characters, so the output stays meaningful.

How It Works (In Simple Steps)
Prepare the Text: The poems are loaded from a .txt file and cleaned up.

Tokenize the Text: Each character is turned into a number (like a language the model understands).

Train the Model: The model reads thousands of small text blocks and learns to predict the next character.

Generate Poems: After training, you can give the model a starting point (like a letter or word), and it will continue writing the poem.

Optional Rhyme Helper: A simple tool checks which words rhyme with a given one and can influence the model’s output.

What You Can Do With It
Generate short poems that sound like Rumi’s work

Explore how neural networks can imitate writing styles

Learn how Transformers and attention mechanisms work

Experiment with rhyme, temperature, and sampling effects

