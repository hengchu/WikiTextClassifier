# Locating Knowledge from Coarse to Fine: An Exploration of DBpedia Ontology

<center><img src="https://goo.gl/u7J5r3" width="400"></center>

Text classification is an old, classical problem in the field of *supervised* machine learning, maybe too old. It is widely used either commercially: building spam filters, depression detection in social media etc., or as a benchmark task for evaluating newly-designed models in academia.

Is text classification a solved task? A general belief is no. What is left there, and how can we improve it? In this article, we will discuss our hypotheses and findings of text classification on DBpedia data.

## Why DBpedia?
**TL; DR**: DBpedia provides categories for entities in different granularity levels. This is awesome for text classification tasks, as we can avoid manual annotation.

<!--<center><img src="http://i66.tinypic.com/2uqis9g.png" width="500"></center>-->
Being one of the most famous parts of the decentralized Linked Data effort (commented by Tim Berners-Lee), [**DBpedia**](https://en.wikipedia.org/wiki/DBpedia) is a project aiming to extract critical structured content from the information created in Wikipedia.

Downloaded from DBpedia website, the data we use is in quad-turtle (tql) serialization. Once decompressed, the dataset contains one line for each queadruple of `<dbpedia_object_url, dbpedia_ontology_info, text, wikipedia_url>`. For example, the first two lines of this file are (text abbreviated):
1. <http://dbpedia.org/resource/Animalia_(book)> <http://dbpedia.org/ontology/abstract>"Animalia is an illustrated children's book..."@en<http://en.wikipedia.org/wiki/Animalia_(book)?oldid=741600610>2. <http://dbpedia.org/resource/List_of_Atlas_Shrugged_characters><http://dbpedia.org/ontology/abstract> "This is a list of characters in Ayn Rand's..."@en<http://en.wikipedia.org/wiki/List_of_Atlas_Shrugged_characters?oldid=744468068>
The `dbpedia_object_url` contains categories that this body of text belongs to.  <br/> For example, the first body of text belongs to these categories:
1. dbc:1986_books2. dbc:Alphabet_books3. dbc:Australian_children's_books4. dbc:Children's_picture_books5. dbc:Picture_books_by_Graeme_Base6. dbc:Puzzle_books

Each of these categories inhabit a level of granularity in a hierarchy. For example,dbc:Puzzle_books is a subcategory of dbc:Puzzles. 


## Performance of different models
Linear Regression

LSTM

Self-attention based model
## Error analysis
## Can two types of categories improve each other?
### Coarse -> Fine
### Fine -> Coarse
## Wrapping up