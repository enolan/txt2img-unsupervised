# Text-to-Image without text

This is research code investigating a concept I call "Embedding Guided Content Generation" (EGCG).
In the broadest terms, EGCG is about generating content in a domain by conditioning generation on
the content's embeddding being within a certain region of embedding space. This repo focuses on
text-to-image generation, where our domain is images and our embedding model is CLIP. EGCG, combined
with the fact that CLIP embeds both text captions and images into a shared embedding space, lets us
build a text-to-image model without using any text in the training data. The pipeline is to first
compute the CLIP text embedding of your prompt, then generate images whose image embeddings are
within a spherical cap centered on the prompt's embedding (equivalently within a certain cosine
distance).

I've tried various approaches for getting EGCG to actually work. [This blog post](http://www.echonolan.net/posts/2024-03-09-is-it-possible-to-train-a-text-to-image-model-without-any-text.html)
shows results from an early attempt, with a single model that takes as input a specification of a
spherical cap and generates images. This doesn't work as well as one would hope. My later work
focuses on a two-stage pipeline, where the first model generates embeddings and the second model
generates images conditioned on those embeddings. There's a *very* early draft of a paper based on
that approach in the `paper` directory. (Ignore the parts about "backwards-forwards importance
sampling", that ended up not working well.)

## Imgur dataset

This repo also contains my code for downloading and processing the ArchiveTeam Imgur dump into a
useable ML dataset. [Another blog post](https://www.echonolan.net/posts/2024-02-21-25m-imgur-images-dataset.html)
describes the dataset. The code is in `process_imgur.py`.