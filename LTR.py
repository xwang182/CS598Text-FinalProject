from collections import deque
import random
import sys
import time
import matplotlib.pyplot as plt

def readData(data,path):
    with open(path) as f:
        content = f.readlines()
    docNum = len(content)
    for line in content:
        arr = line.split()

        doc = []
        relevance = int(arr[0])
        query = int(arr[1].split(":")[1])
        doc.append(query)

        doc.append(float(arr[111].split(":")[1]))
        doc.append(float(arr[116].split(":")[1]))
        doc.append(float(arr[121].split(":")[1]))
        doc.append(float(arr[126].split(":")[1]))

        doc.append(relevance)
        # print doc
        if query in data:
            data[query].append(doc)
        else:
            data[query] = []
            data[query].append(doc)
    return docNum

def allSampling(train):
    output = []
    for qid in train.keys():
        relevanceArr = train[qid]
        relevanceDict = dict()
        relevanceDict[0] = []
        relevanceDict[1] = []
        relevanceDict[2] = []
        relevanceDict[3] = []
        relevanceDict[4] = []
        for one in relevanceArr:
            index = one[5]
            relevanceDict[index].append(one)
        for i in range(0,4):
            for j in range(i+1,5):
                group1 = relevanceDict[i]
                group2 = relevanceDict[j]
                if len(group1) == 0 or len(group2) == 0:
                    continue
                for doc1 in group1:
                    for doc2 in group2:
                        pair = [doc1,doc2]
                        output.append(pair)
    return output

def compare(doc1,doc2):
    compare = [0,0,0,0,-1.0]
    docNum1 = doc1[1:5]
    docNum2 = doc2[1:5]
    for i in range(0,4):
        compare[i] = docNum1[i] - docNum2[i]
    return compare


def docScore(weight,compareArr):
    score = 0.0
    for i in range(0,5):
        score = score + weight[i] * compareArr[i]
    return score


def update_weight(weight,rele,compareArr,rate):
    for i in range(0,5):
        weight[i] = weight[i] + rate * rele * compareArr[i]     #w=w+ryx


def weightInit():
    init = 100.0
    weight = [init,init,init,init,init]
    return weight 


def UnindexSampling(train,number):
    pairs = []
    trainSample = dict()
    trainArr = []
    for i in train.keys():
        trainOne = dict()
        trainOne[0] = []
        trainOne[1] = []
        trainOne[2] = []
        trainOne[3] = []
        trainOne[4] = []
        for one in train[i]:
            for add in range(0,5):
                if add == one[-1]:
                    continue
                else:
                    trainOne[add].append(one)
            trainArr.append(one)
        trainSample[i] = trainOne
    totalLen = len(trainArr)
    while len(pairs) < number:
        rand_one = random.randrange(0, totalLen)
        rand_one = trainArr[rand_one]
        queryId = rand_one[0]
        level = rand_one[-1]
        rand_pool = trainSample[queryId][level]
        if len(rand_pool) == 0:
            continue
        index2 = random.randrange(0, len(rand_pool))
        rand_two = rand_pool[index2]
        pair = [rand_one,rand_two]
        pairs.append(pair)
    return pairs


def IndexSampling(train,number):
    pairs = []
    trainSample = dict()
    for i in train.keys():
        trainOne = dict()
        trainOne[0] = []
        trainOne[1] = []
        trainOne[2] = []
        trainOne[3] = []
        trainOne[4] = []
        trainOne["all"] = []
        for one in train[i]:
            for add in range(0,5):
                if add == one[-1]:
                    continue
                else:
                    trainOne[add].append(one)
            trainOne["all"].append(one)
        trainSample[i] = trainOne
    query_arr = trainSample.keys()
    totalLen = len(query_arr)
    while len(pairs) < number:
        index_query = random.randrange(0, totalLen)
        index_query = query_arr[index_query]
        rand_query = trainSample[index_query]
        index1 = random.randrange(0, len(rand_query["all"]))
        rand_one = rand_query["all"][index1]
        level = rand_one[-1]
        rand_pool = trainSample[index_query][level]
        if len(rand_pool) == 0:
            continue
        index2 = random.randrange(0, len(rand_pool))
        rand_two = rand_pool[index2]
        pair = [rand_one,rand_two]
        pairs.append(pair)
    return pairs

def learn(train,weight,sampling,number):
    rate = 1.0
    if sampling == "no":
        pairs = allSampling(train)
    elif sampling == "unindex":
        pairs = UnindexSampling(train,number)
    elif sampling == "index":
        pairs = IndexSampling(train,number)
    else:
        print "Wrong input format"

    for onepair in pairs:
        doc1 = onepair[0]
        doc2 = onepair[1]

        rele = doc1[-1] - doc2[-1]          #truth label  y
        compareArr = compare(doc1,doc2)     #doc1 - doc2  x   feature vector
        score = docScore(weight,compareArr) #make prediction W*x
        decide = rele * score               #compare with y, ywx
        if decide>0.0:
            continue
        else:
            update_weight(weight,rele,compareArr,rate)
    return weight


def test(testData,trainW):
    correct = 0.0
    total = 0.0
    weight = trainW
    pairs = allSampling(testData)
    for onepair in pairs:
        doc1 = onepair[0]
        doc2 = onepair[1]

        rele = doc1[-1] - doc2[-1]
        compareArr = compare(doc1,doc2)
        score = docScore(weight,compareArr)
        decide = rele * score               #make prediction, compare with y
        if decide>0:
            correct = correct + 1.0
        total = total + 1.0
    accuracy = correct / total * 100.0
    return accuracy

# doc is array of socres [BM25, LMIR.ABS, LMIR.DIR, LMIR.JM, -1]
def oneDoc(weight,doc):
    score = 0.0
    for i in range(0,5):
        score = score + weight[i] * doc[i]
    return score


def approximate(train,docNum):
    keyNum = len(train.keys())
    oneKey = int(docNum / keyNum)
    oneLevel = int(oneKey / 5)
    pairsNum = 10 * oneLevel * oneLevel * keyNum
    return pairsNum


def drawGraph():
    train = dict()
    trainPath = "./train.txt"
    docNum = readData(train,trainPath)
    pairLen = approximate(train,docNum)
    
    testData = dict()
    testPath = "./test.txt"
    readData(testData,testPath)


    number_lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # number_lst = [1,5,10,20,30,40,50,60,70,80,90,100]
    sampling = "index"
    time_lst = []
    acc_lst = []
    for number in number_lst:
        print number
        number = int(number * 0.01 * pairLen)

        avg_time = 0.0
        avg_acc = 0.0
        for i in range(0,3):
            trainW = weightInit()
            startTime = time.time()
            trainW = learn(train,trainW,sampling,number)
            endTime = time.time()
            timeUsed = endTime-startTime
            avg_time = avg_time + timeUsed
            accuracy = test(testData,trainW)
            avg_acc = avg_acc + accuracy
        time_save = avg_time / 3.0
        acc = avg_acc / 3.0
        time_lst.append(time_save)
        acc_lst.append(acc)


    print number_lst
    print time_lst
    print acc_lst
    fig = plt.figure(1)
    if sampling == "index":
        plt.title('Index Sampling percentage against time')
    else:
        plt.title('Unindex Sampling percentage against time')
    plt.xlabel('Sampling Percentage')
    plt.ylabel('Time')
    plt.plot(number_lst, time_lst,linewidth=3,linestyle='solid')
    plt.show()

    fig2 = plt.figure(1)
    plt.title('Index Sampling percentage against accuracy')
    plt.xlabel('Sampling Percentage')
    plt.ylabel('Accuracy')
    plt.plot(number_lst, acc_lst,linewidth=3,linestyle='solid')
    plt.show()




drawGraph()


# train = dict()
# trainPath = "./train.txt"
# docNum = readData(train,trainPath)

# pairLen = approximate(train,docNum)

# sampling = "all"  "unindex" "index"
# sampling = "index"
# sampling = raw_input('Choose sampling method from index sampling, unindex sampling, no sampling\nPlease type "index" or "unindex" or "no": ')
# if sampling != "no":
#     number = raw_input('How many sampling pairs do you want 0-100%\nPlease type a integer 1-100: ')
#     number = int(int(number) * 0.01 * pairLen)

# trainW = weightInit()
# startTime = time.time()
# trainW = learn(train,trainW,sampling,number)
# endTime = time.time()
# print "Weight vector:"
# print trainW
# print sampling + "sampling Training Time:   " + str(endTime-startTime)

# testData = dict()
# testPath = "./test.txt"
# readData(testData,testPath)
# accuracy = test(testData,trainW)
# print "accuracy : " + str(accuracy)





# trainW = [1.0, 0.0, 0.0, 0.0, 0.0]
# testData = dict()
# testPath = "./test.txt"
# readData(testData,testPath)
# accuracy = test(testData,trainW)
# print "BM25 accuracy : " + str(accuracy)

# trainW = [0.0, 1.0, 0.0, 0.0, 0.0]
# testData = dict()
# testPath = "./test.txt"
# readData(testData,testPath)
# accuracy = test(testData,trainW)
# print "ABS accuracy : " + str(accuracy)

# trainW = [0.0, 0.0, 1.0, 0.0, 0.0]
# testData = dict()
# testPath = "./test.txt"
# readData(testData,testPath)
# accuracy = test(testData,trainW)
# print "DIR accuracy : " + str(accuracy)

# trainW = [0.0, 0.0, 0.0, 1.0, 0.0]
# testData = dict()
# testPath = "./test.txt"
# readData(testData,testPath)
# accuracy = test(testData,trainW)
# print "JM accuracy : " + str(accuracy)





