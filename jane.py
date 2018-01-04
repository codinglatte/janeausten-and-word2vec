from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re

documents = ["Human machine interface for lab abc computer applications",
"A survey of user opinion of computer system response time",
"The EPS user interface management system",
"System and human system engineering testing of EPS",
"Relation of user perceived response time to error measurement",
"The generation of random binary unordered trees",
"The intersection graph of paths in trees",
"Graph minors IV Widths of trees and well quasi ordering",
"Graph minors A survey",
"Cats are lovely creatures."]

print("I am here")

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))

stop = set(stopwords.words('english'))
sentence = "this is a foo bar sentence"
cleaned_sentences=[]
for sentence_i in range(0, len(documents)):
	sentence = documents[sentence_i]
	tokenizer = RegexpTokenizer(r'\w+')# matches any word
	letters_only = re.sub(r"\p{P}+", "", sentence) #re.sub(ur"\p{P}+", " ", doc)
	letters_only = letters_only.lower()
	tokens = tokenizer.tokenize(letters_only)
	lemmatized_sentence =[]
	for word_i in range(0, len(tokens)):
		lemmatized_sentence.append(lemmatizer.lemmatize(tokens[word_i]))
	cleaned_sentence = [i for i in lemmatized_sentence if i not in stop]
	cleaned_sentences.append(cleaned_sentence)

	
print(cleaned_sentences)

from sklearn.decomposition import PCA # TSNE
from matplotlib import pyplot	

# train model
model = Word2Vec(cleaned_sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()


# tsne
from sklearn.manifold import TSNE
n_sne = 7000
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

# loading sima txt

# nevekkel - neveket abrazolva


