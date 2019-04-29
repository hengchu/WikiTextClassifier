# Locating Knowledge from Coarse to Fine: An Exploration of DBpedia Ontology

<center><img src="https://goo.gl/u7J5r3" width="400"></center>

Text classification is an old, classical problem in the field of *supervised* machine learning--maybe too old. It is widely used either commercially: building spam filters, depression detection in social media etc., or as a benchmark task for evaluating newly-designed models in academia.

Is text classification a solved task? The general belief is that it is not. What is left there, and how can we improve it? In this article, we will discuss our hypotheses for and findings from text classification on DBpedia data.

## Why DBpedia?
**TL; DR**: DBpedia provides categories for entities at different levels of granularity. This is awesome for text classification tasks, as we can avoid manual annotation for their topics. 

<!--<center><img src="http://i66.tinypic.com/2uqis9g.png" width="500"></center>-->
Being one of the most famous parts of the decentralized Linked Data effort (commented by Tim Berners-Lee), [**DBpedia**](https://en.wikipedia.org/wiki/DBpedia) is a project aiming to extract critical structured content from the information created in Wikipedia.

Downloaded from DBpedia website, the data we use is long abstracts of entities in quad-turtle (tql) serialization. Once decompressed, the dataset contains one line for each quadruple of `<dbpedia_object_url, dbpedia_ontology_info, text, wikipedia_url>`. For example, the first two lines of this file are (text abbreviated):

1. <http://dbpedia.org/resource/Animalia_(book)> <http://dbpedia.org/ontology/abstract>
"Animalia is an illustrated children's book..."@en
<http://en.wikipedia.org/wiki/Animalia_(book)?oldid=741600610>

2. <http://dbpedia.org/resource/List_of_Atlas_Shrugged_characters>
<http://dbpedia.org/ontology/abstract> "This is a list of characters in Ayn Rand's..."@en
<http://en.wikipedia.org/wiki/List_of_Atlas_Shrugged_characters?oldid=744468068>

The `dbpedia_object_url` contains categories that this body of text belongs to.  <br/> For example, the first body of text belongs to these categories:

1. [dbc:1986_books](http://dbpedia.org/page/Category:1986_books)
2. [dbc:Alphabet_books](http://dbpedia.org/page/Category:Alphabet_books)
3. [dbc:Australian\_children's_books](http://dbpedia.org/page/Category:Australian_children's_books)
4. [dbc:Children's\_picture_books](http://dbpedia.org/page/Category:Children's_picture_books)
5. [dbc:Picture\_books\_by\_Graeme_Base](http://dbpedia.org/page/Category:Picture_books_by_Graeme_Base)
6. [dbc:Puzzle_books](http://dbpedia.org/page/Category:Puzzle_books)

The categories above each inhabit a level of granularity in a hierarchy. For example, the `dbpedia_object_url` [dbc:Puzzle_books](http://dbpedia.org/page/Category:Puzzle_books) is a subcategory of [dbc:Puzzles](http://dbpedia.org/page/Category:Puzzles).

This looks great! Those categories serve as perfect labels for our task, and we get away with manual annotation, which is painfully inefficient and expensive.
<img align="right" width="400" src="https://www.researchgate.net/profile/David_Chen136/publication/273122359/figure/fig17/AS:613998374436864@1523400028040/DBpedia-ontology.png">

In our experiments, we extract two types of categories for each piece of text: a **coarse** one and a **fine-grained** one.

The original DBPedia long abstract dataset mapped each body of text to its most
descriptive category, which resulted in more than 1 million total categories. We
used DBPedia's hierarchy dataset on categories, and built a hierarchy graph of
all categories. As we go towards the "top" of this graph, the labels become more
and more abstract, and the total number of labels decrease with respect to how
abstract the labels are. For each original descriptive label, we traversed a
fixed <math><mn>N</mn></math> steps toward the top of the graph to generate a fine label, and a fixed
<math><mn>N+2</mn></math> steps towards the top to generate a coarse label. This reduces the total
amount of labels to a more manageable degree. However, we must be careful to not
reduce the labels too much. Otherwise, a text descibing a music album will get
labeled as "Primates" (because music albums are created by humans, and humans
are primates). We tuned <math><mn>N</mn></math> so that the labels still appear descriptive of the
text for humans.

We finally end up 370 fine categories and 180 coarse categories.




## Performance of different models
### Models
Let's get down to the task, and get some numbers. We implement the task on three different models: **Logistic Regression**, **LSTM**, and a **self-attention** model. These correspond to a non-deep learning benchmark, a base deep model, and an advanced deep model. Hypothetically, the performance will increase with the level of complication of the model. Before jumping to the result, let me expand a bit on the self-attention model.

The self-attention model was inspired by the Transformer model. Generally, a Transformer consists of an encoder and a decoder to transduce sequences. We retain a similar encoder with multi-head self-attention, and then directly add a fully connected linear layer for classification. Our final model has 6 encoder layers, with one typical layer shown in the figure below.

<center><img src="http://jalammar.github.io/images/t/transformer_resideual_layer_norm.png" width="500"></center>
<center> source: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) </center>

### Word Representation
<img align="right" width="200" src="https://i2.wp.com/mlexplained.com/wp-content/uploads/2019/01/bert.png?w=400&ssl=1">
There are various types of pretrained word embeddings out there. However, instead of using these, we decide to train our own embeddings. Specifically, we start with a dictionary that contains thousands of words. Then, we feed the size of vocabulary into `nn.embedding` module in PyTorch and it randomly initializes embeddings. As we train our model, the word embeddings are also trained as a by-product of the learning process.

But which dictionary do we consult? First, we tried to build our own. We took our dataset and filtered out stop words, stripped punctuation, lowercased all words, and threw out all words that occurred less than <math><mn>N</mn></math> times (where <math><mn>N</mn></math> varies from 10 to 100), etc. But when we used our own vocabulary, our LSTM did not perform better than 35%, so we tried using the [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) vocabulary with over 30,000 tokens (which was also used by BERT for tokenization), which increased the accuracy by a LOT!

### Results
To no one's surprise, the advanced deep model outperforms the other two models.
<center>

|                     | Logistic Regression  | LSTM  | Self-attention model |
|:-------------------:|:---------------------:|:-----:| :-------------------:|
|coarse category      | 32.12%                | 43.71%| **44.22%**           |
|fine-grained category| 33.73%                | 42.55%| **43.25%**           |
</center>

For those of you interested in hyperparameters, we set the dimension of word embeddings to be **50** across all experiments. On our self-attention model, increasing the size of embeddings helped improve the accuracy no more than 0.3%, but it takes notoriously longer time to train. So, we kept the dimensions short.

## Error analysis
However, the performance of our more complicated model did not exceed that of the basic LSTM by much. What prevented our model from learning better? Taking a look at some of the wrong predictions that our model made might give us some insight.

Here is an example for fine-grained categorization:

**Raw text**

	ichat Inc (sometimes written iChat Inc) was a company that created instant
	messaging software and associated technology for embedding chat rooms in web
	pages. The company was founded by Andrew Busey. ichat was also the name of
	their initial product. Claims that Apple's iChat client was based on the
	company's technology appear unfounded.

**Coarse category**: Articles</br>
**Fine-grained category**: History</br>
**Top predictions**:
> Information\_technology</br>
> Communication</br>
> Main\_topic\_classifications</br>
> Eponymous_categories</br>
> Business

OK, though *information technology* is arguably a better categorization, we ended up being wrong because *history*, our gold label, is no doubt a more vague and less-related category for a piece of text describing a tech company.

We surmise that this error resulted because we went too far up the classification graph when generating the data category labels (especially for fine-grained ones). When we go too far up the classification graph, the categories become too general to make sense. For example, an article on music albums will eventually be grouped under ‘Human’, ‘Primates’ etc. Therefore, with "super" gold data with labels that are not too fine-grained nor too general, we might achieve better accuracies using current models.

## Can two types of categories improve each other?
Intuitively, coarse labels are easier for models to acquire. We thought that a network for fine-grained classification will converge more quickly if it is pre-trained on coarse labels. Compared to entirely training directly on fine labels, this method is quick, handy, and may be able to achieve comparable results.

On the other hand, we thought that if a network is first trained on finer-grained categories, testing on coarse labels should be a relatively easier task and thus it should perform better. These ideas direct us to the following formal hypotheses, which we validate with experimental results.

### Coarse -> Fine
#### Hypothesis
If we want to train a network to classify the articles into fine categories, we can get improved performance and faster convergence by pre-training on the coarse categories (as compared to directly training on the fine categories).

#### Experiments
We first train an encoder and a classifier on coarsely-labeled data for 2 epochs. We then take this network, swap out the logits layer, and continue training on fine-labeled data. Note that we freeze downstream layers from the logits layer for the first ~100 iterations to make sure that the large gradients caused by reinitializing the logits layer do not mess up the initial pretrained layers. After training, we will evaluate performance and the number of epochs required to achieve performance comparable to baselines.

As a result, using this "**bootstrapping**" method, we achieve a comparable accuracy by continue training on fine labels only **6** epochs, compared to directly training on fine labels for **10** epochs in baseline. We continue training and after 10 epochs, the final accuracy is **43.91%**. A more straightforward comparison is shown in tables below.

<center>
	
#### \# epochs to achieve comparable accuracy

|          | baseline | bootstrapping       |
|:--------:|:--------:|:-------------------:|
| # epochs | 10(fine) | 2(coarse) + 6(fine) |

#### Accuracy after training for 10 epochs on fine labels

|          | baseline | bootstrapping |
|:--------:|:--------:|:-------------:|
| accuracy | 43.25%   | **43.91%**    |

</center>

### Fine -> Coarse
#### Hypothesis
Starting with a network trained to classify the articles into fine categories, we can generate a new network able to classify the articles into coarser categories by simply retraining the last few layers.

#### Experiments
We train an encoder and a classifier on fine-labeled data. Then we keep this encoder, but swap the logits layer and retrain the model on coarse labels. The performance is compared to baselines in the table below. Both models are trained for 10 epochs. We observe a substantial increase in the performance of a model pre-trained on fine-grained categories.
<center>

|          | baseline | pre-trained model |
|:--------:|:--------:|:-----------------:|
| accuracy | 44.22%   | **49.05%**        |
</center>

## Wrapping up
We have investigated the classical problem in NLP, text classification. The experiments present the capacity and effectiveness of neural-based models such as LSTM. Moreover, attention mechanisms are powerful at extending the capacity of neural models. The next wave of language-based learning algorithms promises even greater abilities such as learning with limited labeled data.

**Notice**: Thanks if you made it this far. Are there errors you would love to correct? Feel free to leave comments and share your thoughts. We also put together our code on [github](https://github.com/hengchu/cis700project).
