# Language modelling

The exercise shows how a language model may be used to solve word-prediction tasks and to generate text.


## Tasks

Objectives (8 points):

1. Read the documentation of [Language modelling in the Transformers](https://huggingface.co/transformers/task_summary.html#language-modeling) library.
2. Download three [Polish models](https://huggingface.co/models?filter=pl) from the Huggingface repository. These should be regular language models, which were not fine-tuned. E.g. `HerBERT` and `papuGaPT2` are good examples. You can also try using Bielik for that, but make sure you are using the model via Transformers API, not GUI.
3. Devise a method to test if the langage model understands Polish cases. E.g. testing for *nominal case* could be expressed as "Warszawa to największe `[MASK]`", and the masked word should be in nominative case. Create sentences for each case.
4. Devise a method to test long-range relationships such as gender. E.e. you can use two verbs with masculine and feminine gender, where one of the verbs is masked. Both verbs should have the same gender, assuming the subject is the same. Define at least 3 such sentences.
5. Check if the model captures real-world knolwedge. For instance a sentence "`[MASK]` wrze w temperaturze 100 stopni, a zamarza w temperaturze 0 stopni Celsjusza." checks if the model "knows" the description of water. Define at least 3 such sentences.
6. Check zero-shot learning capabilites of the models. Provide at least 5 sentences with different sentiment for the following scheme: "'Ten film to był kiler. Nie mogłem się oderwać od ekranu.' Wypowiedź ta ma jest zdecydowanie `[MASK]`" Try different prompts, to see if they make any difference.
7. Take into accout the fact, that causal language models such as PapuGaPT2 or plT5, will only generate continuations of the sentenes, so the examples have to be created according to that paradigm.


Answer the following questions (2 points):
   1. Which of the models produced the best results?
   2. Was any of the models able to capture Polish grammar?
   3. Was any of the models able to capture long-distant relationships between the words?
   4. Was any of the models able to capture world knowledge?
   5. Was any of the models good at doing zero-shot classification?
   6. What are the most striking errors made by the models?

## Hints

1. Language modelling (LM) is a task concentrated on computing the probability distribution of words given a sequence of
   preceding words.
2. In the past the most popular LM were based on n-gram counting. The distribution of probability of the next word was
   approximated by the relative frequency of the last n-words, preceding this word. Usually n=3, since larger values
   resulted in extremely large datasets.
3. Many algorithms were devised to improve that probability estimates for infrequent words. Among them Kneser-Ney was
   the most popular.
4. SRI LM is the most popular toolkit for creating traditional language models.
5. At present recurrent neural networks, attention networks and transformers are the most popular neural-network
   architectures for creating LMs.
6. The largest LM currently is GPT-3 described in (mind the number of authors!) *Language Models are Few-Shot Learners*
   Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav
   Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon
   Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler,
   Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
   Sutskever, Dario Amodei
