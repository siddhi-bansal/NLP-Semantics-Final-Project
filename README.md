## Visualization Details  

The repository contains visualizations organized under the [`/laser`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/laser), [`/labse`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/labse), and [`/openai`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/openai) directories.  

### Heatmaps  
Located in the `/heatmaps` subdirectory [`laser heatmaps`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/laser/heatmaps), [`labse heatmaps`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/labse/heatmaps), [`openai heatmaps`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/openai/heatmaps)), these visualizations display cosine similarities across all sentences (not binned).  
- Each heatmap compares one language to a baseline language (specified in the file title).  
- The `all_languages_similarity_heatmap.png` provides an overview of universal similarity across all languages and sentences.  

### Wordclouds  
Available in the `/wordclouds` subdirectory [`laser wordclouds`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/laser/wordclouds), [`labse wordclouds`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/labse/wordclouds), [`openai wordclouds`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/openai/wordclouds), these wordclouds depict cosine similarities between each language and all other languages.  

### Sentence Complexity Analysis (LASER Model)  
We performed clustering of heatmaps and wordclouds based on sentence types (simple, complex, and compound) specifically for the LASER model, our baseline. These are located in the 
[`/laser/complexity_and_similarity`](https://github.com/siddhi-bansal/NLP-Semantics-Final-Project/tree/main/laser/complexity_and_similarity) directory.  
- The `all_similarities_heatmap.png` highlights universally similar languages, focusing on specific sentence complexity categories (simple, compound, or complex).  
- The wordclouds here represent cosine similarities of languages, restricted to sentences of the same complexity category.  
