Analysis plan:

I plan on using a topic model approach to find underlying factors in the clinical notes.
Specifically, I will use LDA (latent Dirichlet allocation) to find topics common to sets of the documents, which will provide insights into the sort of problems each patient faces.
I will use Gensim for building the model, as well as spaCy for tokenization and other pre-processing.
I will combine both the text of the notes as well as the named entities into one corpus, and use features like bigrams as well.
Afterwards, I will use a coherence score to do hyperparameter tuning, specifically, the alpha and beta parameters, as well as the number of topics to create.
There seems to be a good package for visualizing topics, pyLDAvis, so I will use that as well.

Post-development notes:

I found a good guide that does similar things to what I wanted, so I mostly followed it, but had to adjust and change some things.
(https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0)
I did some parameter tuning and found that generally, an alpha value closer to 0 and a beta value closer to 1 produced better coherence scores. This makes sense, because patients generally will only have a couple factors (or topics) relevant to them, which would mean a lower alpha value is better at modeling the notes. And, since many words can be used to describe similar problems, a higher beta value is better as well.
(Alpha corresponds to document-topic density, while beta corresponds to topic-word density).
I had some trouble figuring out an optimal number of topics, however. As seen in coherence_topics.png, where I ran several models with the same number of topics and took their average coherence score, one would think that 16-18 topics was optimal. When I was looking at the words for each topic produced by these models, it was very inconsistent.
Sometimes there were very good topics about issues as specific as being attacked by a dog, but other times many of the topics had similar words in common that didn't provide much helpful information. In visualization.html, I used pyLDAvis for a model with 9 topics.
I tried getting a model that had the better topics as I just described, but unfortunately was unable to reproduce it.
I probably could have used the random_state attribute to avoid this issue in the future.