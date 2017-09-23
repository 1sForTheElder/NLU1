# coding: utf-8

import gensim
import math
from copy import copy
import re
import time
from gensim.models import ldamodel

'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''

class Load_Corpus_with_Iteration(object): # 使用迭代器读取语料库
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line.split()
class BncSentences:
	def __init__(self, corpus, n=-1):
		self.corpus = corpus
		self.n = n
	
	def __iter__(self):
		n = self.n
		ret = []
		for line in open(self.corpus):
			line = line.strip().lower()
			if line.startswith("<s "):
				ret = []
			elif line.strip() == "</s>":
				if n > 0:
					n -= 1
				if n == 0:
					break
				yield copy(ret)
			else:
				parts = line.split("\t")
				if len(parts) == 3:
					word = parts[-1]
					idx = word.rfind("-")
					word, pos = word[:idx], word[idx+1:]
					if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
						pos = "r"
					if pos == "j":
						pos = "a"
					ret.append(gensim.utils.any2unicode(word + "." + pos))

'''
(a) function load_corpus to read a corpus from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile, contextFile):
	id2word = {}
	word2id = {}
	vectors = []
	wd = open(vocabFile)
	fp = open(contextFile);
	for i in fp:
		i = re.sub('\n','',i)
		i = re.split('\t | |:',i);
		del i[0]
		# print len(i)
		matrix1 = []
		if len(i) <= 1:
			vectors.append([])
			continue
		else:
			for g in range(0,len(i)):

				if g%2 == 0 and g<len(i):

				# print g
				# if g == 0:
					# print i
					matrix1.append((int(i[g]),int(i[g+1])))

			vectors.append(matrix1)

	n = 0
	for i in wd:
		i = re.sub('\n','',i)
		word2id[i] = n
		id2word[n] = i
		n+=1


	# print len(vectors)
	# print vectors[21928]
	# your code here
	# print word2id,id2word
	# print vectors[21926]
	# print vectors[6784]
	# print '6786',vectors[6785]
	# print '6787',vectors[6786]
	# print vectors[19999]
	print len(vectors)
	return id2word, word2id, vectors
# load_corpus("vocabulary.txt","word_contexts.txt")
'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
# def cosine_similarity2(vector1, vector2):
# 	# your code here
# 	print 'hahaha'
#
# 	cos_sim = 0
# 	if type(vector1[0]) == str:
# 		print vector1, vector2
# 		up = 0
# 		v1 = 0
# 		v2 = 0
# 		for i in len(vector1):
# 			up += vector1[i]*vector2[i]
# 			v1 += vector1[i]**2
# 			v2 += vector2[i]**2
# 		cos_sim = up/(math.sqrt(v1)*math.sqrt(v2))
# 	print cos_sim
# 	return cos_sim
def cosine_similarity(vector1, vector2):
	# your code here
	# if type(vector1[0]) == str:
	# 	print vector1, vector2
	# 	up = 0
	# 	v1 = 0
	# 	v2 = 0
	# 	for i in range(0,len(vector1)):
	# 		# print 'hjahaha'
	# 		up += float(vector1[i])*float(vector2[i])
	# 		v1 += float(vector1[i])**2
	# 		v2 += float(vector2[i])**2
	# 	print cos_sim
	# 	cos_sim = up/(math.sqrt(v1)*math.sqrt(v2))
	if type(vector1[0]) == tuple:
		# print 'hahaha'
		# print vector1[-1][0],vector2[-1][0]
		g = 0
		gg = 0
		for b in vector2:
			g = max(g,b[0])
		for a in vector1:
			gg = max(gg,a[0])

		maxCol =  (max(g,gg)+1)
		# maxColl = ([max(y[0],x[0]) for x,y in zip(vector1,vector2)])
		vv1 = [0]*maxCol
		vv2 = [0]*maxCol
		# vv2 = [0]*(max(vector1[-1][0]+100,vector2[-1][0]))
		# print len(vv1)
		for i in vector1:
			# print i[0]
			vv1[i[0]] = str(i[1])
		for g in vector2:
			# print g[0],len(vv2)
			vv2[g[0]] = str(g[1])
		cos_sim = cosine_similarity(vv1,vv2)
	else:
		up = 0
		v1 = 0
		v2 = 0
		for i in range(0,len(vector1)):
            #
			# print vector2
			# print vector2[i]
			up += float(vector1[i])*float(vector2[i])
			v1 += float(vector1[i])**2
			v2 += float(vector2[i])**2
		cos_sim = up/(math.sqrt(v1)*math.sqrt(v2))
	return cos_sim

'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
	start = time.time()
	tfIdfVectors = []
	if type(freqVectors[0][0]) == tuple:
		full_vector = []
		for i in freqVectors:
			part_of_vector = [0]*5000
			for ii in i:
				part_of_vector[ii[0]] = ii[1]
			full_vector.append(part_of_vector)
		tfIdfVectors = tf_idf(full_vector)
		return tfIdfVectors
	N = len(freqVectors)
	context_len = len(freqVectors[0])
	vv = []
	for i in range(0,context_len):
		time_num = 0.0
		for ii in freqVectors:
			if ii[i]>0:
				time_num += 1
		if time_num>0:
			vv.append(1+math.log(N/time_num,2))
		else:
			vv.append(0)
	print ';timmm' , time.time()-start
	for i in freqVectors:
		raw_volumn = []
		for a in range(0,context_len):
			if i[a]>0:
				if math.log(i[a],2)>1:
					raw_volumn.append((1+math.log(i[a],2))*vv[a])
			else: raw_volumn.append(0)
		tfIdfVectors.append(raw_volumn)
	fin = time.time()
	print 'time:',fin-start
	return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):
	# your code here
	sen = BncSentences(corpus)
	# senn = Load_Corpus_with_Iteration(corpus)
	# print type(sen),type(senn)
	model = []
	# for i in sen:
	# 	print i
	# sentences = []
	# ff = BncSentences.__init__(corpus,5)
	# print ff
	# sentence = open(sen.corpus)

	# for i in sentence.readlines():
    #
	# 	i.strip()
	# 	i = re.split('\s',i)
	# 	i.pop()
	# 	sentences.append(i)
    #
	model = gensim.models.Word2Vec(sen, size=100, window=5, min_count=5, workers=8,alpha=learningRate,sample=downsampleRate,negative=negSampling)
	print "finish one time"
	# gensim.models.Word2Vec
	# gensim.models.Word2Vec
	# print model
	return model

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping):
	# your code here
	a = gensim.models.ldamodel.LdaModel(corpus=vectors,id2word=wordMapping,update_every=0,passes=10)
	return a

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID):
	# your code here
	wordids = []
	listt = ldaModel[topicID]
	for i in listt:
		wordids.append(i)
	return words

if __name__ == '__main__':
	import sys
	part = sys.argv[1].lower()
	
	# these are indices for house, home and time in the data. Don't change.
	house_noun = 80
	home_noun = 143
	time_noun = 12
	
	# this can give you an indication whether part a (loading a corpus) works.
	# not guaranteed that everything works.
	if part == "a":
		print("(a): load corpus")
		try:
			id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
			if not id2word:
				print("\tError: id2word is None or empty")
				exit()
			if not word2id:
				print("\tError: id2word is None or empty")
				exit()
			if not vectors:
				print("\tError: id2word is None or empty")
				exit()
			print("\tPass: load corpus from file")
		except Exception as e:
			print("\tError: could not load corpus from disk")
			print(e)

		try:
			if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
				print("\tError: id2word fails to retrive correct words for ids")
			else:
				print("\tPass: id2word")
		except Exception:
			print("\tError: Exception in id2word")
			print(e)
		
		try:
			if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
				print("\tError: word2id fails to retrive correct ids for words")
			else:
				print("\tPass: word2id")
		except Exception:
			print("\tError: Exception in word2id")
			print(e)
	
	# this can give you an indication whether part b (cosine similarity) works.
	# these are very simple dummy vectors, no guarantee it works for our actual vectors.
	if part == "b":
		import numpy
		print("(b): cosine similarity")
		try:
			cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
			if not numpy.isclose(0.5, cos):
				print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: sparse vector similarity")
		except Exception:
			print("\tError: failed for sparse vector")
		try:
			cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
			if not numpy.isclose(0.5, cos):
				print("\tError: full expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: full vector similarity")
		except Exception:
			print("\tError: failed for full vector")

	# you may complete this part to get answers for part c (similarity in frequency space)
	if part == "c":
		id_word, word_id, word_context_vector = load_corpus(sys.argv[2], sys.argv[3]);
		# print c
		# print word_context_vector[word_id["house.n"]]
		# print word_context_vector[word_id["home.n"]]
		cos1 = cosine_similarity(word_context_vector[word_id["house.n"]], word_context_vector[word_id["home.n"]])
		cos= cosine_similarity(word_context_vector[word_id["house.n"]], word_context_vector[word_id["time.n"]])
		cos2 = cosine_similarity(word_context_vector[word_id["home.n"]], word_context_vector[word_id["time.n"]])
		print("(c) similarity of house, home and time in frequency space",cos,cos1,cos2)
		
		# your code here
	
	# this gives you an indication whether your conversion into tf-idf space works.
	# this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
	if part == "d":
		print("(d) converting to tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		try:
			tfIdfSpace = tf_idf(vectors)
			if not len(vectors) == len(tfIdfSpace):
				print("\tError: tf-idf space does not correspond to original vector space")
			else:
				print("\tPass: converted to tf-idf space")
		except Exception as e:
			print("\tError: could not convert to tf-idf space")
			print(e)
	
	# you may complete this part to get answers for part e (similarity in tf-idf space)
	if part == "e":
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		# print vectors[word2id["house.n"]]
		tf_vec = tf_idf(vectors)
		start1 = time.time()
		# vec_house_n = []
		# vec_home_n = []
		# vec_time_n = []
		# for r in tf_vec:
		# 	vec_home_n.append(r[word2id["home.n"]])
		# 	vec_house_n.append(r[word2id["house.n"]])
		# 	vec_time_n.append(r[word2id["time.n"]])

		# tf_vec =  [[r[col] for r in tf_vec] for col in range(len(tf_vec[0]))]
		print time.time()-start1
		coss = cosine_similarity(tf_vec[word2id["house.n"]],tf_vec[word2id["home.n"]]);
		coss2 = cosine_similarity(tf_vec[word2id["time.n"]], tf_vec[word2id["home.n"]]);
		coss3 = cosine_similarity(tf_vec[word2id["house.n"]], tf_vec[word2id["time.n"]]);
		# coss = cosine_similarity(vec_home_n,vec_house_n)
		# coss2 = cosine_similarity(vec_home_n,vec_time_n)
		print("(e) similarity of house, home and time in tf-idf space",coss,coss2,coss3)
		
		# your code here
	
	# you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
	if part == "f1":

		leanringRatee = [0.005*i+0.01 for i in range(9)]
		downsamplingRate = [0.1*10**(-i) for i in range(5)]
		negativeRate = [i for i in range(0,11)]
		# inpAccu = []
		ff = open('accs5.txt', 'w')
		for i in leanringRatee:
			for ii in downsamplingRate:
				for iii in negativeRate:
					# print i, ii, iii
					# i = round(i,2)
					# ii = round(ii,2)

					f = word2vec("bnc.vert",i,ii,iii)
					o = f.accuracy("accuracy_test.txt")
					correctNum = 0.0
					incorrectNum = 0.0
					for i1 in o:
						correctNum += (len(i1['correct']))
						incorrectNum += len((i1['incorrect']))
					if (correctNum + incorrectNum) > 0:
						accuracy = float(correctNum) / (correctNum + incorrectNum)
					else:
						accuracy = 0

					# print "caonimabi",f["paris.n"]
					inputt = 'learningRate:' + str(i) + 'downsamplingRate:' + str(ii) + 'negativeRate:' + str(iii) + "   =:   " + str(accuracy)
					print type(inputt)
					print inputt
					ff.write(inputt)
					ff.write('\n')


		# f = word2vec("bnc.vert", 0.01, 0.1, 5)
        #
		# o =  f.accuracy("accuracy_test.txt")
		# correctNum = 0.0
		# incorrectNum = 0.0
		# for i in o:
		# 	correctNum+=(len(i['correct']))
		# 	incorrectNum+=len((i['incorrect']))
		# # if (float(correctNum)+float(incorrectNum) > 0:
		# if (correctNum+incorrectNum) >0:
		# 	accuracy = float(correctNum)/(correctNum+incorrectNum)
		# else: accuracy = 0
        #
		# inp = []
		# inputt = 'learningRate:'+ str(leanringRate)+'downsamplingRate:'+str(downsamplingRate)+'negativeRate:'+str(negativeRate)+ "   =:   "+str(accuracy)
		# inp.append(inputt)
		# print accuracy
		# o1 = f.log_accuracy(o)
		# print o1



		# ff = open('testaccuracy.txt','w')
		# # print "caonimabi",f["paris.n"]
		# for key in inputt:
		# 	# print key
		# 	ff.write(key)
		# 	ff.write('\n')



			# ff.write(i)
		# print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling",f)
		
		# your code here
	
	# you may complete this part for the second part of f (training and saving the actual word2vec model)
	if part == "f2":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		Word2Vec = word2vec("bnc.vert", 0.05, 0.00125, 9)
		gensim.models.Word2Vec.load('word2vec.model')
		o = Word2Vec.accuracy("accuracy_test.txt")
		correctNum = 0.0
		incorrectNum = 0.0
		for i1 in o:
			correctNum += (len(i1['correct']))
			incorrectNum += len((i1['incorrect']))
		if (correctNum + incorrectNum) > 0:
			accuracy = float(correctNum) / (correctNum + incorrectNum)
		else:
			accuracy = 0
		# print "caonimabi",f["paris.n"]
		inputt = str(accuracy)
		print("(f2) word2vec, building full model with best parameters. May take a while.")
		
		# your code here
	
	# you may complete this part to get answers for part g (similarity in your word2vec model)
	if part == "g":

		w2v_house_home_sim = cosine_similarity(Word2Vec[word2id["home.n"]],Word2Vec[word2id["house.n"]])
		w2v_house_time_sim = cosine_similarity(Word2Vec[word2id["time.n"]],Word2Vec[word2id["house.n"]])
		w2v_time_home_sim  = cosine_similarity(Word2Vec[word2id["home.n"]],Word2Vec[word2id["time.n"]])
		print("(g): word2vec based similarity")
		
		# your code here
	
	# you may complete this for part h (training and saving the LDA model)
	if part == "h":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		id2word,word2id,vectors = load_corpus(sys.argv[2],sys.argv[3])
		LDA = lda(vectors,id2word)
		LDA.save("lda")
		print("(h) LDA model",LDA)
		
		# your code here
	
	# you may complete this part to get answers for part i (similarity in your LDA model)
	if part == "i":
		print("(i): lda-based similarity")
		# # your code here
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		LDA = gensim.models.LdaModel.load("lda.model")
		print cosine_similarity(LDA[vectors[word2id['home.n']]],LDA[vectors[word2id['house.n']]])
		print cosine_similarity(LDA[vectors[word2id['home.n']]], LDA[vectors[word2id['time.n']]])
		print cosine_similarity(LDA[vectors[word2id['have.v']]], LDA[vectors[word2id['own.v']]])

	# you may complete this part to get answers for part j (topic words in your LDA model)
	if part == "j":
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		topic = sys.argv[4]
		LDA = lda(vectors, id2word)
		words = get_topic_words(LDA,topic)
		wordlist = []
		for i in words:
			wordlist.append(id2word(i))
		print("(j) get topics from LDA model",wordlist)
		
		# your code here
