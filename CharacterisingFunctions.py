#taking the code out of CharacteristicWords.ipynb and analysis.ipynb to make available elsewhere

import matplotlib.pyplot as plt
import numpy as np
import operator
import math




# For a given set of corpora, find the frequency distribution of the k highest frequency words
# Output total size of corpus and sorted list of term, frequency pairs

def find_hfw_dist(corpora, k=100000):
    # add worddicts for individual corpora
    # sort and output highest frequency words
    # visualise

    sumdict = {}
    corpussize = 0
    for acorpus in corpora:
        for (key, value) in acorpus.allworddict.items():
            sumdict[key.lower()] = sumdict.get(key.lower(), 0) + value
            corpussize += value

    print("Size of corpus is {}".format(corpussize))
    candidates = sorted(sumdict.items(), key=operator.itemgetter(1), reverse=True)
    # print(candidates[:50])
    # print(len(sumdict))
    # print(sumdict)
    return corpussize, candidates[:k]


def makedict(alist):
    adict = {}
    for (key, value) in alist:
        adict[key] = adict.get(key, 0) + value
    return adict


def pmi(wordfreq, refwordfreq, corpussize, refcorpussize):
    if wordfreq * refcorpussize * refwordfreq * corpussize == 0:
        score = 0
    # print(wordfreq,refwordfreq,corpussize,refcorpussize)
    else:
        score = np.log((wordfreq * refcorpussize) / (refwordfreq * corpussize))
    return score


def rev_pmi(wordfreq, refwordfreq, corpussize, refcorpussize):
    return pmi(refwordfreq - wordfreq, refwordfreq, refcorpussize - corpussize, refcorpussize)


def llr(wordfreq, refwordfreq, corpussize, refcorpussize):
    # print(wordfreq,refwordfreq,corpussize,refcorpussize)
    mypmi = pmi(wordfreq, refwordfreq, corpussize, refcorpussize)
    myrevpmi = rev_pmi(wordfreq, refwordfreq, corpussize, refcorpussize)
    # myrevpmi2=rev_pmi2(wordfreq,refwordfreq,corpussize,refcorpussize)
    # print(mypmi,myrevpmi,myrevpmi2)
    llr_score = 2 * (wordfreq * mypmi + (refwordfreq - wordfreq) * myrevpmi)
    if pmi(wordfreq, refwordfreq, corpussize, refcorpussize) < 0:
        return -llr_score
    else:
        return llr_score


def klp(p, q):
    return p * np.log((2 * p) / (p + q))


def kl(wordfreq, refwordfreq, corpussize, refcorpussize):
    # ref should be the total corpus - function works out difference

    p = wordfreq / corpussize
    q = (refwordfreq - wordfreq) / (refcorpussize - corpussize)

    return klp(p, q)


def jsd(wordfreq, refwordfreq, corpussize, refcorpussize):
    p = wordfreq / corpussize
    q = (refwordfreq - wordfreq) / (refcorpussize - corpussize)

    k1 = klp(p, q)
    k2 = klp(q, p)
    score = 0.5 * (k1 + k2)
    if p > q:
        return score
    else:
        return -score


def likelihoodlift(wordfreq, refwordfreq, corpussize, refcorpussize, alpha):
    beta = 0
    if alpha == 1:
        return math.log(wordfreq / corpussize)
    elif alpha == 0:
        return pmi(wordfreq, refwordfreq, corpussize, refcorpussize)
    else:
        return (alpha * math.log(beta + (wordfreq / corpussize)) + (1 - alpha) * pmi(wordfreq, refwordfreq, corpussize,
                                                                                     refcorpussize))


def mysurprise(wf, rwf, cs, rcs, measure, params):
    if measure == 'pmi':
        return pmi(wf, rwf, cs, rcs)
    elif measure == 'llr':
        return llr(wf, rwf, cs, rcs)
    elif measure == 'kl':
        return kl(wf, rwf, cs, rcs)
    elif measure == 'jsd':
        return jsd(wf, rwf, cs, rcs)
    elif measure == 'likelihoodlift':
        return likelihoodlift(wf, rwf, cs, rcs, params.get('alpha', 0.5))
    else:
        print("Unknown measure of surprise")


def improved_compute_surprises(corpusA, corpusB, measure, params={},k=50,display=True):
    (corpusAsize, wordlistA) = corpusA
    (corpusBsize, wordlistB) = corpusB
    if 'threshold' in params.keys():
        threshold = params['threshold']
    else:
        threshold = len(wordlistA)
    # dictA=makedict(wordlistA)
    dictB = makedict(wordlistB)

    scores = []
    # print(wordlistA[:threshold])
    for (term, freq) in wordlistA[:threshold]:
        scores.append((term, mysurprise(freq, dictB.get(term, freq + 1), corpusAsize, corpusBsize, measure, params)))
    sortedscores = sorted(scores, key=operator.itemgetter(1), reverse=True)
    if display and k>0:
        print("Top {} terms are ".format(k))
        print(sortedscores[:k])
    rank = 0
    if measure == "llr":
        for (term, score) in sortedscores:
            if score > 10.828:
                rank += 1
            else:
                break
        print("{} significantly characterising terms".format(rank))
    else:
        rank = k
    return (sortedscores[:rank])


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """

    maxheight=np.array([rect.get_height() for rect in rects]).max()
    if maxheight>1:
        aformat='%1.1f'
        add=math.log(maxheight,10)
    else:
        aformat='%.3f'
        add=0.0005

    #print(maxheight,aformat)
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + add,
                aformat % height,
                ha='center', va='bottom')
    return (maxheight+add)*1.1

def display_list(hfw_list,cutoff=10,words=[],leg=None,title=None,ylim=10,abbrevx=True,xlabel='High Frequency Words',ylabel='Probability',colors=None):
    width=0.7/len(hfw_list)
    toplot=[]
    for hfw in hfw_list:
        corpussize=hfw[0]
        if words==[]:
            todisplay=hfw[1][:cutoff]
        else:
            todisplay=[(x,y) for (x,y) in hfw[1] if x in words]
            cutoff=len(words)
        barvalues=sorted(todisplay,key=operator.itemgetter(0),reverse=False)
        #print(barvalues)
        xs,ys=[*zip(*barvalues)]
        if corpussize>0:
            ps=[y*100/corpussize for y in ys]
        else:
            ps=ys

        toplot.append(ps)

    #print(toplot)
    N=len(xs)
    ind=np.arange(N)
    fig,ax=plt.subplots(figsize=(2*cutoff,cutoff/2))
    rectset=[]
    if colors==None:
        colors=['r','b','y','g']
    for i,ps in enumerate(toplot):
        rectset.append(ax.bar(ind+i*width,ps,width,color=colors[i]))
    
    if leg!=None:
        ax.legend(rectset,leg)
    ax.set_xticks(ind)
    if abbrevx:
        xs=[x.split(' ')[0] for x in xs]
    ax.set_xticklabels(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for rects in rectset:
        ylim=autolabel(rects,ax)
    if title!=None:
        ax.set_title(title)
    ax.set_ylim(0,ylim)


    return xs
    


def improved_display_list(xvalues, yvalueslist, labels={}):
    width = 0.7 / len(yvalueslist)
    N = len(xvalues)
    ind = np.arange(N)
    fig, ax = plt.subplots(figsize=(20, 12))
    rectset = []
    colors = ['r', 'b', 'y', 'g']
    for i, ps in enumerate(yvalueslist):
        rectset.append(ax.bar(ind + i * width, ps, width, color=colors[i]))

    leg = labels.get('leg', None)
    title = labels.get('title', None)
    xlabel = labels.get('xlabel', 'Year')
    ylabel = labels.get('ylabel', 'Probability')
    ylim = labels.get('ylim', 1)
    if leg != None:
        ax.legend(rectset, leg)
    ax.set_xticks(ind)
    ax.set_xticklabels(xvalues)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim)
    for rects in rectset:
        autolabel(rects, ax)
    if title != None:
        ax.set_title(title)
    plt.show()


# We have a corpus e.g., male_corpus and a set of characterising terms for that corpus e.g., malewords
def find_pos(term, corpus):
    pospos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PUNCT', 'PROPN']
    counts = {}
    for apos in pospos:
        counts[apos] = corpus.wordposdict.get((term, apos), 0)

    total = sum(counts.values())

    gt = corpus.allworddict.get(term, 0)
    counts['OTHER'] = gt - total
    # print(term,gt,counts)
    if gt > 0:
        poses = [(tag, weight / gt) for (tag, weight) in counts.items()]
    else:
        poses = []
    # print(term,poses)
    return poses


def analyse(termset, corpus):
    freqs = []
    somefreqs = []
    posdict = {}
    someposdict = {}
    threshold = 20
    for i, (term, relevance) in enumerate(termset):
        freq = corpus.allworddict[term]
        freqs.append(freq)
        if i < threshold:
            somefreqs.append(freq)
        poses = find_pos(term, corpus)
        for mypos, weight in poses:
            posdict[mypos] = posdict.get(mypos, 0) + weight
            if i < threshold:
                someposdict[mypos] = someposdict.get(mypos, 0) + weight

    freqarray = np.array(freqs)
    meanfreq = np.mean(freqarray)
    sdfreq = np.std(freqarray)
    meanprob = meanfreq / corpus.wordtotal
    sdprob = sdfreq / corpus.wordtotal
    print("Mean frequency is {}, sd is {}".format(meanfreq, sdfreq))
    print("Mean probability is {}, sd is {}".format(meanprob, sdprob))
    somefreqarray = np.array(somefreqs)
    meansomefreq = np.mean(somefreqarray)
    sdsomefreq = np.std(somefreqarray)
    meansomeprob = meansomefreq / corpus.wordtotal
    sdsomeprob = sdsomefreq / corpus.wordtotal
    print("For top {} words, mean freq is {}, sd is {}".format(threshold, meansomefreq, sdsomefreq))
    print("For top {} words, mean prob is {}, sd is {}".format(threshold, meansomeprob, sdsomeprob))
    # print(posdict)
    xvalues = posdict.keys()
    totaly = sum(posdict.values())
    totalz = sum(someposdict.values())
    allvalues = []
    somevalues = []
    for x in xvalues:
        allvalues.append(posdict.get(x, 0))
        somevalues.append(someposdict.get(x, 0))
    yvalues = [[100 * y / totaly for y in allvalues], [100 * z / totalz for z in somevalues]]
    labels = {'title': 'Distribution of POS in Characterising Terms', 'xlabel': 'Part of Speech',
              'ylabel': 'Proportion', 'leg': ['Whole Set', "Top {}-restricted Set".format(threshold)], 'ylim': 100}
    improved_display_list(xvalues, yvalues, labels)


def nearest_neighbours(wordset, w2vmodel):
    threshold = 20
    found = 0
    for i, (term, score) in enumerate(wordset):
        try:
            neighbours = w2vmodel.wv.most_similar([term])
            found += 1
            if i < threshold:
                print(term, neighbours)
        except:
            print("{} not in vocab".format(term))

    oov = 100 - (found * 100 / len(wordset))
    print("Out of vocabulary: {}".format(oov))


def make_matrix(wordset, model, threshold=0.5):
    matrix = []

    for (termA, _score) in wordset:
        row = []
        for (termB, _score) in wordset:
            try:
                sim = model.wv.similarity(termA, termB)
                if sim < threshold:
                    sim = 0
            except:
                sim = 0
            row.append(sim)

        matrix.append(row)
    return matrix

punctdict = {"\n": "_NEWLINE", ";": "_SEMICOLON", ":": "_COLON", "\"": "_QUOTE", "'s": "_GEN", "-": "_HYPHEN",
             "(": "_LEFTBRACKET", ")": "_RIGHTBRACKET", ",": "_COMMA", ".": "_FULLSTOP", "..": "_DOTDOT"}


def clean(term):
    # remove punctuation which will confuse Gephi
    cleanterm = punctdict.get(term, term)
    return cleanterm


def make_csv(wordset, model, filename, threshold=0.5):
    matrix = make_matrix(wordset, model, threshold=threshold)
    terms = [clean(term) for (term, score) in wordset]

    # with open(filename,'w') as csvfile:
    #    csvwriter=csv.writer(csvfile,dialect='excel')
    #    headings=['']+terms
    # print(headings)
    #    csvwriter.writerow(headings)
    #    for term,row in zip(terms,matrix):
    #        csvwriter.writerow([term]+row)

    with open(filename, 'w') as csvfile:
        line = ""
        for term in terms:
            line += ';' + term
        line += '\n'

        csvfile.write(line)
        # print(line)
        for term, row in zip(terms, matrix):
            line = term
            # print(row)
            for item in row:
                line += ';' + str(item)

            line += '\n'

            csvfile.write(line)
            # print(line)


def find_topk(alist, k):
    # ignore top neighbour as this is the word itself

    sortedlist = sorted(alist, reverse=True)
    if sortedlist[1] == 0:
        return []
    if k == -1:
        return (sortedlist[1:])
    else:
        return (sortedlist[1:k + 1])


def semantic_coherance(word_set, model, k=1, verbose=True):
    matrix = make_matrix(word_set, model)
    # print(matrix)
    mysum = 0
    total = 0
    for row in matrix:
        topk = find_topk(row, k)
        mysum += sum(topk)
        total += len(topk)
    if total == 0:
        average = 0
    else:
        average = mysum / total
    if verbose:
        print("Average semantic coherance at k={}: {}".format(k, average))
    return average


def coherance_profile(words, model, verbose=True):
    scores = []
    scores.append(semantic_coherance(words, model, k=1, verbose=verbose))
    scores.append(semantic_coherance(words, model, k=2, verbose=verbose))
    scores.append(semantic_coherance(words, model, k=5, verbose=verbose))
    scores.append(semantic_coherance(words, model, k=10, verbose=verbose))
    scores.append(semantic_coherance(words, model, k=-1, verbose=verbose))
    return scores


def frequency_profile(wordsets, corpus, labels=[]):
    allfrequencies = []
    for wordset in wordsets:
        frequencies = []
        # print(wordset)
        for (word, score) in wordset:
            frequencies.append(int(corpus.allworddict[word]))
        allfrequencies.append(np.array(frequencies))
    # print(allfrequencies)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.boxplot(allfrequencies, showmeans=True, labels=labels)
    ax.set_title('Frequency Profile of Characteristic Words')
    ax.set_yscale('log')
    plt.show()


ft = 25

def frequency_threshold(csets, threshold=ft, corpus=None):
    tsets = []
    for cset in csets:
        tset = []
        for term, score in cset:
            freq = int(corpus.allworddict[term])
            if freq > threshold:
                tset.append((term, score))
        tsets.append(tset)

    return tsets


if __name__=="__main__":

    print("This file contains functions taken from CharacterisingWords.ipynb")
    print("No tests have been written, as yet, in this file")
