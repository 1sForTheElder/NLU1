# coding: utf-8

from question1 import *
import json
import gensim
from gensim.models import Word2Vec

class JSObject:
    def __init__(self, d):
         self.__dict__ = d

def make_vector(sparsetuple,dimen=0):
	if type(sparsetuple[0])!=tuple:
		return sparsetuple
	len = max(max([x[0] for x in sparsetuple]),(dimen-1))+1
	newvector = [0]*len
	for eles in sparsetuple:
		newvector[eles[0]] = eles[1]
	return newvector

'''
helper class to load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
	thesaurus = {}
	with open(thesaurusFile) as inFile:
		for line in inFile.readlines():
			word, subs = line.strip().split("\t")
			thesaurus[word] = subs.split(" ")
	return thesaurus

'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
	# your code here
	if type(vector1[0]) == tuple:
		maxlen = (max(vector1[-1][0], vector2[-1][0]) + 1)
		vec1 = make_vector(vector1,maxlen)
		vec2 = make_vector(vector2,maxlen)
		return addition(vec1,vec2)
	else:
		add_vec = []
		for i in range(len(vector1)):
			if vector1[i]+vector2[i]>0:
				add_vec.append((i,vector1[i]+vector2[i]))
	return add_vec

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):

	if type(vector1[0]) == tuple:
		maxlen = (max(vector1[-1][0], vector2[-1][0]) + 1)
		vec1 = [0] * maxlen
		vec2 = [0] * maxlen
		for i in vector1:
			vec1[i[0]] = i[1]
		for ii in vector2:
			vec2[ii[0]] = ii[1]
		return multiplication(vec1,vec2)
	else:
		mul_num = []
		for i in range(len(vector1)):
			if vector1[i]*vector2[i]>0:
				mul_num.append((i,vector1[i]*vector2[i]))
	# your code here
	return mul_num

'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
	# your code here
	topic_prob = 0.0
	sum_pro = 0.0
	for i in ldaModel.get_document_topics(wordVector,0):
		if topicID == i[0]:
			topic_prob = i[1]
		sum_pro += i[1]
	topic_prob = topic_prob/sum_pro
	return topic_prob

'''
(d) function prob_w_given_z to get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
def prob_w_given_z(ldaModel, targetWord, topicID):
	# your code here
	term_prob = 0.0
	sum_pro = 0.0
	for i in ldaModel.get_topic_terms(topicID,20000):
		if word2id[targetWord] == i[0]:
			term_prob = i[1]
		sum_pro += i[1]
	term_prob = term_prob/sum_pro
	return term_prob

def calculate_prob_list_of_tc(t_word,c_word,ldaModel,wordVector):
	prob_list = []
	sum = 0
	for i in range(0,100):
		upper_left = prob_z_given_w(ldaModel,i,wordVector[word2id[t_word]])
		upper_right = prob_w_given_z(ldaModel,c_word,i)
		prob_list.append(upper_left*upper_right)
		sum+=(upper_left*upper_right)
	# print prob_list
	return prob_list,sum

'''
(f) get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
input: jsonSentence, a string in json format
input: thesaurus, mapping from target words to candidate substitution words
input: word2id, mapping from vocabulary words to word IDs
input: model, a vector space, Word2Vec or LDA model
input: frequency vectors, original frequency vectors (for querying LDA model)
input: csType, a string indicating the method of calculating context sensitive vectors: "addition", "multiplication", or "lda"
output: the best substitution word for the jsonSentence in the given model, using the given csType
'''
def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType):
	# (b) use addition to get context sensitive vectors
	if csType == "addition":
		modeltype = ''
		for content_sens in jsonSentence:
			content_sens = json.loads(content_sens,object_hook=JSObject)
			target_w = (content_sens.target_word).encode('utf-8')
			target_pos = int((content_sens.target_position).encode('utf-8'))
			target_id = int((content_sens.id).encode('utf-8'))
			target_sen = (content_sens.sentence).encode('utf-8')
			target_sen = re.split('\s*',target_sen)
			scores = {}
			for sub_words in thesaurus[target_w]:
				try:
					vector_of_the_sub_word = model[word2id[sub_words]]
					modeltype = "tf_idf_"
				except:
					try:
						vector_of_the_sub_word = model[sub_words]
						modeltype = "Word2Vec"
					except:
						continue
				score = 0.0
				for ind in range(target_pos - 5, target_pos + 6):
					vec_t_c = []
					if ind == target_pos or ind < 0:
						pass
					else:
						try:
							vec_t_c = addition(model[word2id[target_sen[ind]]], model[word2id[target_w]])
						except:
							try:
								vec_t_c = addition(model[target_sen[ind]], model[target_w])
							except:
								continue
					if len(vec_t_c) > 0:
						vec_t_c = make_vector(vec_t_c, len(vector_of_the_sub_word))
						score += cosine_similarity(vector_of_the_sub_word, vec_t_c)
				scores[sub_words] = score
			best_word = ''
			best_score = -100
			for sc in scores:
				if scores[sc] > best_score:
					best_word = sc
					best_score = scores[sc]
			filename = csType+modeltype+".txt"
			outing = open(filename, 'a')
			best_word = best_word[0:-2]
			outputthings = str(target_w) + " " + str(target_id) + ' :: ' + str(best_word)
			outing.write(outputthings)
			outing.write('\n')
			print outputthings
		# your code here
		outing.close()
		pass
		
	# (c) use multiplication to get context sensitive vectors
	elif csType == "multiplication":
		for content_sens in jsonSentence:
			# print 'memeda'
			content_sens = json.loads(content_sens,object_hook=JSObject)
			target_w = (content_sens.target_word).encode('utf-8')
			target_pos = int((content_sens.target_position).encode('utf-8'))
			target_id = int((content_sens.id).encode('utf-8'))
			target_sen = (content_sens.sentence).encode('utf-8')
			target_sen = re.split('\s*',target_sen)
			scores = {}
			for sub_words in thesaurus[target_w]:
				try:
					vector_of_the_sub_word = model[word2id[sub_words]]
					modeltype = "tf_idf_"
				except:
					try:
						vector_of_the_sub_word = model[sub_words]
						modeltype = "Word2Vec"
					except:
						continue
				score = 0.0
				for ind in range(target_pos - 5, target_pos + 6):
					vec_t_c = []
					if ind == target_pos or ind < 0:
						pass
					else:
						try:
							vec_t_c = multiplication(model[word2id[target_sen[ind]]], model[word2id[target_w]])
						except:
							try:
								vec_t_c = multiplication(model[target_sen[ind]], model[target_w])
							except:
								continue

					if len(vec_t_c) > 0:
						vec_t_c = make_vector(vec_t_c, len(vector_of_the_sub_word))
						score += cosine_similarity(vector_of_the_sub_word, vec_t_c)
				scores[sub_words] = score
			best_word = ''
			best_score = -100
			for sc in scores:
				if scores[sc]>best_score:
					best_word = sc
					best_score = scores[sc]
			filename = modeltype+csType+'.txt'
			print "目标词:", target_w, '\n', "分数组：", scores,'ID',target_id
			outing = open(filename,'a')
			best_word = best_word[0:-2]
			outputthings = str(target_w)+" "+str(target_id)+' :: '+str(best_word)
			print outputthings
			outing.write(outputthings)
			outing.write('\n')
		# your code here
		outing.close()
		pass
		
	# (d) use LDA to get context sensitive vectors
	elif csType == "lda":
		# your code here
		for content_sens in jsonSentence:
			content_sens = json.loads(content_sens,object_hook=JSObject)
			target_w = (content_sens.target_word).encode('utf-8')
			target_pos = int((content_sens.target_position).encode('utf-8'))
			target_id = int((content_sens.id).encode('utf-8'))
			target_sen = (content_sens.sentence).encode('utf-8')
			target_sen = re.split('\s*',target_sen)
			scores = {}
			for sub_words in thesaurus[target_w]:
				try:
					vector_of_the_sub_word = [0]*100
					vector_of_the_sub_word_1 = model.get_term_topics(word2id[sub_words],0)
					if len(vector_of_the_sub_word_1)<1:
						continue
					for i in vector_of_the_sub_word_1:
						vector_of_the_sub_word[i[0]] = i[1]
				except:
					continue
				score = 0.0
				for ind in range(target_pos - 5, target_pos + 6):
					vec_t_c = []
					if ind == target_pos or ind < 0:
						pass
					else:
						try:
							prob_l,prodt = calculate_prob_list_of_tc(target_sen[ind], target_w, ldaModel, vectors)
							for i in prob_l:
								vec_t_c.append(i/prodt)
						except:
							continue
					if len(vec_t_c) > 0:
						vec_t_c = make_vector(vec_t_c, len(vector_of_the_sub_word))
						score += cosine_similarity(vector_of_the_sub_word, vec_t_c)
				scores[sub_words] = score
			best_word = ''
			best_score = -100
			for sc in scores:
				if scores[sc]>best_score:
					best_word = sc
					best_score = scores[sc]
			best_word = best_word[0:-2]
			filename = 'output_lda.txt'
			outing = open(filename,'a')
			outputthings = str(target_w)+" "+str(target_id)+' :: '+str(best_word)
			print outputthings
			outing.write(outputthings)
			outing.write('\n')
		#your code here
		pass
	
	return None

if __name__ == "__main__":
	import sys
	
	part = sys.argv[1]
	
	# this can give you an indication whether part a (vector addition and multiplication) works.
	if part == "a":
		print("(a): vector addition and multiplication")
		v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
		try:
			if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
				print("\tError: sparse addition returned wrong result")
			else:
				print("\tPass: sparse addition")
		except Exception as e:
			print("\tError: exception raised in sparse addition")
			print(e)
		try:
			if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
				print("\tError: sparse multiplication returned wrong result")
			else:
				print("\tPass: sparse multiplication")
		except Exception as e:
			print("\tError: exception raised in sparse multiplication")
			print(e)
		try:
			addition(v3,v4)
			print("\tPass: full addition")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
		try:
			multiplication(v3,v4)
			print("\tPass: full multiplication")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
	
	# you may complete this to get answers for part b (best substitution words with tf-idf and word2vec, using addition)
	if part == "b":
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		W2V = gensim.models.Word2Vec.load("Word2Vec.model")
		tfIdfSpace = tf_idf(vectors)
		print("(b) using addition to calculate best substitution words")
		# your code here

		sens = open('test.txt')
		thesaurus = load_thesaurus("test_thesaurus.txt")
		tf_add_best_sub = best_substitute(jsonSentence=sens,frequencyVectors=vectors,thesaurus=thesaurus,word2id=word2id,model=tfIdfSpace,csType='addition')
		sens.close()
		sens = open("test.txt")
		w2v_add_best_sub = best_substitute(jsonSentence=sens, frequencyVectors=vectors, thesaurus=thesaurus,word2id=word2id, model=W2V, csType='addition')
		sens.close()

		print ('TF-IDF addition best substitution words')
		print ('Word2vec addition best substitution words')

	# you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
	if part == "c":
		print("(c) using multiplication to calculate best substitution words")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		W2V = gensim.models.Word2Vec.load("Word2Vec.model")
		tfIdfSpace = tf_idf(vectors)
		print("(b) using addition to calculate best substitution words")
		# your code here

		sens = open('test.txt')
		thesaurus = load_thesaurus("test_thesaurus.txt")
		best_substitute(jsonSentence=sens,frequencyVectors=vectors,thesaurus=thesaurus,word2id=word2id,model=tfIdfSpace,csType='multiplication')
		sens.close()
		sens = open('test.txt')
		best_substitute(jsonSentence=sens, frequencyVectors=vectors, thesaurus=thesaurus,word2id=word2id, model=W2V, csType='multiplication')
		sens.close()
	
	# this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
	if part == "d":
		print("(d): calculating P(Z|w) and P(w|Z)")
		print("\tloading corpus")
		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
		print("\tloading LDA model")
		ldaModel = gensim.models.ldamodel.LdaModel.load("lda")
		houseTopic = ldaModel[vectors[word2id["house.n"]]][0][0]
		print ldaModel[vectors[word2id["house.n"]]]
		try:
			if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
				print("\tPass: P(Z|w)")
			else:
				print("\tFail: P(Z|w)")
		except Exception as e:
			print("\tError: exception during P(Z|w)")
			print(e)
		try:
			if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
				print("\tPass: P(w|Z)")
			else:
				print("\tFail: P(w|Z)")
		except Exception as e:
			print("\tError: exception during P(w|Z)")
			print(e)
	
	# you may complete this to get answers for part d2 (best substitution words with LDA)
	if part == "e":
		print("(e): using LDA to calculate best substitution words")
		sens = open('test.txt')
		thesaurus = load_thesaurus("test_thesaurus.txt")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		ldaModel = gensim.models.ldamodel.LdaModel.load("lda")
		best_substitute(jsonSentence=sens, frequencyVectors=vectors, thesaurus=thesaurus, word2id=word2id, model=ldaModel,csType='lda')
